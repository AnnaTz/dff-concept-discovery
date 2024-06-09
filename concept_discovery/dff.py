from math import floor
import numpy as np
import sys
from matplotlib import pyplot as plt
import torch
from torchvision import models, transforms
import os
from .utils import imresize, make_grid
from .nmf import NMF
import cv2
import shutil


# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variable to store features
conv_output = {}

def hook_features(name):
    """
    Create a hook function that captures the output of a specified layer.
    
    Parameters:
        - name (str): The name of the layer whose output is to be captured.
    
    Returns:
        - Function: A hook function.
    """
    def hook(model, input, output):
        conv_output[name] = output.detach()
    return hook


def load_network():
    """
    Load a pre-trained VGG-19 network and modify it for feature extraction.
    
    Returns:
        - model: A modified VGG-19 model.
    """
    net = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(device)
    net.eval()
    layer_name = 'features'
    del net.features._modules['36']  # Remove max-pooling after the final conv layer
    net.features.register_forward_hook(hook_features(layer_name))
    return net, layer_name


# Prepare the network that will provide the features for DFF
net, layer_name = load_network()


def preprocess_images(data_list):
    """
    Preprocess a sublist of images: reading, resizing, and normalizing.
    
    Parameters:
        - data_list (list of str): List of image file paths.
        
    Returns:
        - numpy.ndarray: A batch of preprocessed images.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    raw_images_batch = []
    for filename in data_list:
        img = plt.imread(filename)
        if img.ndim == 2:  # Convert grayscale to RGB
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = transform(img)
        raw_images_batch.append(img)
        
    return torch.stack(raw_images_batch)


def get_features(data_list):
    """
    Extract features for a list of images using the prepared model.
    
    Parameters:
        - data_list (list of str): List of image file paths.
        
    Returns:
        - Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors containing extracted 
        features and raw images.
    """
    batch_size = 100
    n_batches = floor(len(data_list) / batch_size)
    raw_images = []
    features = []

    for batch_index in range(n_batches + 1):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size \
            if batch_index < n_batches \
            else len(data_list)
        data_sublist = data_list[start_index:end_index]

        if not data_sublist:
            continue

        t_raw_images_batch = preprocess_images(data_sublist).to(device)
        raw_images.append(t_raw_images_batch)

        # Forward pass to get features
        with torch.no_grad():
            net(t_raw_images_batch)
            features.append(conv_output[layer_name])

    features = torch.cat(features, 0)
    raw_images = torch.cat(raw_images, 0)
    return features, raw_images


def normalize_heatmaps(heatmaps):
    """
    Normalize heatmaps by their maximum values.
    
    Parameters:
        - heatmaps (torch.Tensor): Tensor containing heatmaps.
        
    Returns:
        - torch.Tensor: Tensor containing the normalized heatmaps.
    """
    return heatmaps / \
        heatmaps.max(axis=3, keepdims=True)[0].max(axis=2, keepdims=True)[0]


def calculate_concept_certainties(norm_concept_heatmaps, orig_concept_heatmaps):
    """
    Calculate concept certainty based on the concept heatmaps.
    
    Parameters:
        - norm_concept_heatmaps (numpy.ndarray): The normalized concept heatmaps.
        - orig_concept_heatmaps (numpy.ndarray): The original concept heatmaps before 
        normalization.
    
    Returns:
        - list: The average certainty across all images for each concept.
    """
    certainties = []
    for v in range(norm_concept_heatmaps.shape[0]):
        norm_heatmap = norm_concept_heatmaps[v]
        orig_heatmap = orig_concept_heatmaps[v]
        
        threshold = 0.1
        concept_mask = norm_heatmap >= threshold
        num_significant_pixels = np.count_nonzero(concept_mask,axis=(1,2))
        
        certainty = np.sum(
            concept_mask*orig_heatmap,axis=(1,2)
        ) / num_significant_pixels
        certainties.append(certainty)
    
    return certainties


def save_plot(ranks, certainties, optimalK, filename='nmf_certainty_curve.png'):
    """
    Save a plot of the mean concept discovery certainty against the number of factors.
    
    Parameters:
        - ranks (list): The list of ranks (number of factors) tested.
        - certainties (list): The list of mean certainties for each rank.
        - optimalK (int): The optimal number of factors determined.
        - filename (str): The filename to save the plot.
    """
    plt.figure()
    plt.plot(ranks, certainties, 'o-')
    plt.xticks(ranks, ranks)
    plt.ylabel('Mean Concept Certainty')
    plt.xlabel('Number of Factors')
    plt.tight_layout()
    plt.axvline(optimalK, color='k', dashes=[2, 2])
    plt.savefig(filename)
    plt.close()


def optimize_concept_count(data_list, make_plot=False):
    """
    Determine the optimal number of factors (concepts)by performing DFF for different 
    numbers and measuring the certainty.
    
    Parameters:
        - data_list (list): List of image file paths.
        - make_plot (bool): Whether to generate a plot of the certainties.
        
    Returns:
        - int: The optimal number of factors.
    """
    features, _ = get_features(data_list)
    flat_features = features \
        .permute(0, 2, 3, 1).contiguous().view((-1, features.size(1)))
    ranks = range(2, 7)
    certainties = []
    
    for K in ranks:
        with torch.no_grad():
            W, _ = NMF(
                flat_features, 
                K, 
                random_seed=0, 
                cuda=features.device.type == 'cuda', 
                max_iter=50,
            )
        concept_heatmaps = W \
            .view(features.size(0), features.size(2), features.size(3), K) \
                .permute(0, 3, 1, 2)
        orig_concept_heatmaps = torch.nn.functional.interpolate(
            concept_heatmaps, size=(224, 224), mode='bilinear', align_corners=False
        )
        normalized_concept_heatmaps = normalize_heatmaps(orig_concept_heatmaps)
        
        concept_certainties = calculate_concept_certainties(
            normalized_concept_heatmaps.cpu().numpy(), 
            orig_concept_heatmaps.cpu().numpy(),
        )
        certainties.append(np.mean(concept_certainties))
    
    optimalK = ranks[np.argmax(certainties)]
    
    if make_plot:
        save_plot(ranks, certainties, optimalK)
    
    return optimalK


def inference_dff(data_list, pre_file):
    """
    Perform inference using DFF to generate heatmaps for given data.
    
    Parameters:
        - data_list (list): List of image file paths.
        - pre_file (str): Path to the file containing the precomputed factors (H).
        
    Returns:
        torch.Tensor: The generated heatmaps.
    """
    try:
        pre_H = torch.from_numpy(np.loadtxt(pre_file, dtype=float)).float()
        if pre_H.dim() == 1:
            pre_H = pre_H.unsqueeze(dim=0)
    except IOError as e:
        print(f'Error reading file {pre_file}: {e}')
        sys.exit()

    features, _ = get_features(data_list)
    flat_features = features \
        .permute(0, 2, 3, 1).contiguous().view((-1, features.size(1)))
    
    with torch.no_grad():
        W, _ = NMF(
            flat_features, 
            pre_H.shape[0], 
            H=pre_H, 
            random_seed=0, 
            cuda=features.device.type == 'cuda', 
            max_iter=50,
        )
    
    heatmaps = W \
        .view(features.size(0), features.size(2), features.size(3), pre_H.shape[0]) \
            .permute(0, 3, 1, 2)
    return torch.nn.functional.interpolate(
        heatmaps, size=(224, 224), mode='bilinear', align_corners=False
    )


def training_dff(K, data_list, class_name):
    """
    Train and save a DFF model for the given class of images, and save a grid of 
    heatmapped images to represent each discovered concept.
    
    Parameters:
        - K (int): The number of factors to use in NMF.
        - data_list (list): List of image file paths.
        - class_name (str): The name of the class to be studied.
    
    Returns:
        - numpy.ndarray: The concept heatmaps generated for the training images.
    """
    results_path = os.path.join('results','training_results',class_name)
    os.makedirs(results_path,exist_ok=True)
    grid_path = os.path.join(results_path,'grids')
    grid_path_short = os.path.join(results_path,'grids_short')
    
    features, raw_images = get_features(data_list)
    flat_features = features \
        .permute(0, 2, 3, 1).contiguous().view((-1, features.size(1)))
    device = features.device

    with torch.no_grad():
        W, H = NMF(
            flat_features, K, random_seed=0, cuda=device.type == 'cuda', max_iter=50
        )

    pre_file = os.path.join(results_path,class_name+'.txt')
    np.savetxt(pre_file, H.cpu(), fmt='%.8e')

    orig_heatmaps = W \
        .view(features.size(0), features.size(2), features.size(3), K) \
            .permute(0, 3, 1, 2)
    orig_heatmaps = torch.nn.functional.interpolate(
        orig_heatmaps, size=(224, 224), mode='bilinear', align_corners=False
    )
    heatmaps = normalize_heatmaps(orig_heatmaps)
    orig_heatmaps = orig_heatmaps.cpu().numpy()
    heatmaps = heatmaps.cpu().numpy()

    process_and_save_heatmaps(
        heatmaps, raw_images, orig_heatmaps, data_list, grid_path, grid_path_short
    )
    
    for i in range(K):
        member_path = os.path.join(grid_path, str(i))
        if os.listdir(member_path):
            grid_name = 'grid_' + str(i) + '.png'
            make_grid(member_path, grid_path, grid_name, (224, 224))
            shutil.rmtree(member_path)

            member_path_short = os.path.join(grid_path_short, str(i))
            make_grid(member_path_short, grid_path_short, grid_name, (224, 224))
            shutil.rmtree(member_path_short)
                
    return orig_heatmaps


def process_and_save_heatmaps(
    heatmaps, raw_images, orig_heatmaps, data_list, grid_path, grid_path_short
):
    """
    Process heatmaps to create images with high activation contours. Create and save 
    grids of these images; a full grid (containing all training images) and a summary
    grid (containing images with the highest concept certainties).

    Parameters:
        - heatmaps (numpy.ndarray): Normalized concept heatmaps.
        - raw_images (torch.Tensor): Tensor of preprocessed raw images.
        - orig_heatmaps (numpy.ndarray): Original concept heatmaps before normalization.
        - data_list (list): List of image file paths.
        - grid_path (str): Path for saving the full grids.
        - grid_path_short (str): Path for saving the summary grids.
    """
    concept_certainties = np.stack(
        calculate_concept_certainties(heatmaps, orig_heatmaps)
    )
    certainty_threshold = np.sort(concept_certainties, axis=0)[-16]
    
    for v, image_path in enumerate(data_list):
        image_name = os.path.basename(image_path)

        for q in range(heatmaps.shape[1]):
            raw_dir = os.path.join(grid_path,str(q))
            os.makedirs(raw_dir,exist_ok=True)
            raw_dir_short = os.path.join(grid_path_short,str(q))
            os.makedirs(raw_dir_short,exist_ok=True)
            
            # Process each heatmap
            mask = heatmaps[v, q] >= 0.75
            _, labels = cv2.connectedComponents(mask.astype(np.uint8))
            for i, unique_label in enumerate(np.unique(labels)):
                if unique_label == 0:
                    continue  # Skip the background
                label_mask = labels == unique_label
                if np.sum(label_mask) < (0.01 * label_mask.size): # Skip small parts
                    continue
                
                # Apply the mask to the original resized image to highlight the concept
                raw_image = raw_images[v].permute(1, 2, 0).cpu().numpy()
                patched =  raw_image * np.stack((label_mask,)*3, axis=2)
                grayscale = cv2.cvtColor(np.uint8(255*patched), cv2.COLOR_BGR2GRAY)
                thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)
                contour, hierarchy = cv2.findContours(
                    thresholded[1],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE
                )
                raw_image_resized = np.uint8(
                    imresize(plt.imread(data_list[v]), 224, 224)*255
                ) 
                highlighted_image = cv2.drawContours(
                    raw_image_resized.copy(), contour, -1, (0, 255, 0), 2
                )

                name_end = image_name.rfind(".")
                part_name = f'{image_name[:name_end]}_p-{q}-{i}{image_name[name_end:]}'
                plt.imsave(os.path.join(raw_dir, part_name), highlighted_image)
                
                if concept_certainties[v,q] > certainty_threshold[q]:
                    plt.imsave(
                        os.path.join(raw_dir_short, part_name), highlighted_image
                    )
