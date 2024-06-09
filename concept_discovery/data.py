import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import models
from .utils import clear_directory, imresize
import shutil
from .dff import get_features
from .nmf import NMF


def load_dataset_classes(filepath='imagenet_data/ImageNet_classes.txt'):
    """
    Load dataset classes from the given text file containing one class name per line. 
    The order of the classes is the same as the order of their logits in the DFF model.
    
    Parameters:
        - filepath (str): The path to the text file.
    
    Returns:
        - list: A list of strings, where each string is a class name.
    """
    try:
        with open(filepath) as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f'Error: The file {filepath} was not found.')
        sys.exit()


def preprocess_images(source_data_path, filenames):
    """
    Load and preprocess images by resizing and normalizing. 
    Only colored images are kept.
    
    Parameters:
        - source_data_path (str): The directory path where the images are stored.
        - filenames (list of str): List of filenames of images within source_data_path 
        to be processed.
    
    Returns:
        - Tensor: A tensor containing the processed images.
    """
    raw_images = []
    for filename in filenames:
        img_path = os.path.join(source_data_path, filename)
        img = plt.imread(img_path)
        if len(img.shape) == 3:  # Check for colour images
            resized_img = imresize(img, 224, 224)
            raw_images.append(resized_img)
    raw_images = np.stack(raw_images)

    # Normalize images
    images = raw_images.transpose((0, 3, 1, 2)).astype('float32')  # NxCxHxW, float32
    images -= np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))  # zero mean
    images /= np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))  # unit variance
    images = torch.from_numpy(images)  # convert to PyTorch tensor

    return images


def filter_images_by_class(
    images, filenames, source_data_path, target_data_path, class_name, DATASET_CLASSES
):
    """
    Copy images to a target directory if they can be correctly classified by DFF's 
    network with high confidence.
    
    Parameters:
        - images (Tensor): A batch of images to consider.
        - filenames (list of str): The filenames corresponding to the images.
        - source_data_path (str): The source directory path of the original images.
        - target_data_path (str): The target directory path for the selected images.
        - class_name (str): The target class name for evaluating the images.
        - DATASET_CLASSES (list): A list of class names indexed like classifier logits.
    """
    cuda = torch.cuda.is_available()
    net = models.vgg19(weights=models.VGG19_Weights.DEFAULT)  # Load pre-trained VGG-19
    if cuda:
        net = net.cuda()
        images = images.cuda()
        
    for i, img in enumerate(images):
        img = img.unsqueeze(dim=0)
        gt_p = torch.nn.functional.softmax(net(img), dim=1)
        gt_c = torch.max(gt_p).item()
        gt_c_i = torch.argmax(gt_p).item()
        gt_c_i_label = DATASET_CLASSES[gt_c_i]
        if ',' in gt_c_i_label:
            gt_c_i_label = gt_c_i_label[:gt_c_i_label.find(",")]

        if gt_c >= 0.95 and gt_c_i_label == class_name:
            shutil.copy(os.path.join(source_data_path, filenames[i]), target_data_path)


def init_train_data(class_name, filter=True):
    """
    Initialize training data for DFF on the given class.
    
    Parameters:
        - nclass_name (str): The name of the class.
        - filter (bool): Whether to filter the available data or not. If True, only 
        images that can be classified with high confidence will be kept.
    """
    source_data_path = os.path.join('imagenet_data', class_name)
    target_data_path = os.path.join('results', 'training_data', f'{class_name}_train')

    clear_directory(target_data_path)
    os.makedirs(target_data_path)

    filenames = os.listdir(source_data_path)

    if not filter:
        for filename in filenames:
            shutil.copy(os.path.join(source_data_path, filename), target_data_path)
    else:
        images = preprocess_images(source_data_path, filenames)
        DATASET_CLASSES = load_dataset_classes()
        filter_images_by_class(
            images, 
            filenames, 
            source_data_path, 
            target_data_path, 
            class_name, 
            DATASET_CLASSES,
        )

        if len(os.listdir(target_data_path)) < 10:
            print('Given data not good enough. Maybe decrease P threshold.')
            sys.exit()


def perform_dff(data_list, K):
    """
    Perform DFF on the given data.
    
    Parameters:
        - data_list (list of str): List of paths to the images that will be analyzed.
        - K (int): The number of factors (concepts) to be discovered.
    
    Returns:
        - tuple: A tuple of two numpy arrays; the first containing the normalized 
        heatmaps produced by DFF and the second containing the original heatmaps before 
        normalization.
    """
    features, _ = get_features(data_list)
    device = features.device

    flat_features = features \
        .permute(0, 2, 3, 1) \
            .contiguous().view((-1, features.size(1)))  # NxCxHxW -> (N*H*W)xC

    with torch.no_grad():
        W, _ = NMF(
            flat_features, 
            K, 
            random_seed=0, 
            cuda=device.type.startswith('cuda'), 
            max_iter=50, 
        )

    heatmaps = W \
        .view(features.size(0), features.size(2), features.size(3), K) \
            .permute(0, 3, 1, 2)  # (N*H*W)xK -> NxKxHxW
    heatmaps = torch.nn.functional.interpolate(
        heatmaps, size=(224, 224), mode='bilinear', align_corners=False
    )  # Resize heatmaps

    orig_heatmaps = heatmaps.detach().clone()
    heatmaps = heatmaps / \
        heatmaps.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    heatmaps = heatmaps.cpu().numpy()
    orig_heatmaps = orig_heatmaps.cpu().numpy()

    return heatmaps, orig_heatmaps


def filter_data_certainty(data_path, K, certainty_threshold=0.1, quantile=0.25):
    """
    Filter data based on the certainty of the concepts DFF can discover in them.
    Images associated with low certainty are removed from the dataset.
    
    Parameters:
        - data_path (str): Path of the directory containing the data to filter.
        - K (int): The number of concepts to be discovered.
        - certainty_threshold (float): Threshold for considering a region in a concept 
        heatmap significant.
        - quantile (float): Quantile to define the threshold for removing images with 
        low certainty.
    """
    temp_data_path = f'{data_path}_temp'
    shutil.copytree(data_path, temp_data_path)

    data_list = [
        os.path.join(temp_data_path, filename) 
        for filename in os.listdir(temp_data_path)
    ]
    heatmaps, orig_heatmaps = perform_dff(data_list, K)

    # Compute mean certainty
    mean_cert = compute_certainty(heatmaps, orig_heatmaps, certainty_threshold)
    thresh = np.quantile(mean_cert, quantile)

    for i, cert in enumerate(mean_cert):
        if cert < thresh:
            os.remove(data_list[i])

    if len(os.listdir(temp_data_path)) >= 15:
        shutil.rmtree(data_path)
        shutil.move(temp_data_path, data_path)
    else:
        shutil.rmtree(temp_data_path)


def compute_certainty(heatmaps, orig_heatmaps, threshold):
    """
    Compute the certainty associated with the provided heatmaps. 
    Certainty is computed as the mean of significant regions' values, and it serves as 
    a measure of how certain the DFF discovery is for the corresponding image.
    
    Parameters:
        - heatmaps (np.array): The normalized concept heatmaps.
        - orig_heatmaps (np.array): The original heatmaps before normalization.
        - threshold (float): The certainty threshold used to define significant regions 
        within the heatmaps.
    
    Returns:
        - list: A list of certainty values for each input heatmap.
    """
    certainties = []
    for i, heatmap in enumerate(heatmaps):
        num = np.count_nonzero(heatmap >= threshold, axis=(1, 2))
        concept_mask = heatmap >= threshold
        cert = np.sum(concept_mask * orig_heatmaps[i], axis=(1, 2)) / num
        certainties.append(np.mean(cert))
    return certainties
