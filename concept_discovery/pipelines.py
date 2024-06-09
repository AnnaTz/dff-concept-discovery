import os
from .dff import training_dff, inference_dff, optimize_concept_count
import matplotlib
from .data import *
from matplotlib import pyplot as plt
import numpy as np
from .utils import imresize, clear_directory


def prepare_data(
    class_name, 
    data_path, 
    prediction_filter=True, 
    certainty_filter=False,
    num_concepts=None, 
):
    """
    Creates the training dataset for concept discovery on the given class. 
    Available images can be filtered out based on given criteria.   

    Parameters:
    - class_name: The name of the class of the training images.
    - data_path: The path where the data is stored.
    - prediction_filter: Whether to filter the data based on classification confidence.
    - certainty_filter: Whether to filter the data based on concept certainty.
    - num_concepts: The number of concepts being discovered. If None, it is estimated.
    
    Returns:
    - A list of paths to the images selected for training.
    """
    init_train_data(class_name, prediction_filter)
    data_list = [
        os.path.join(data_path, filename) for filename in os.listdir(data_path)
    ]
    if num_concepts is None:
        num_concepts = optimize_concept_count(data_list)
        print(num_concepts)
    if certainty_filter:
        filter_data_certainty(data_path, num_concepts)
    return [os.path.join(data_path, filename) for filename in os.listdir(data_path)]


def run_training_pipeline(
    class_names,
    prediction_filter=True,
    certainty_filter=False,
    num_concepts=None,
):
    """
    Executes the training pipeline of concept discovery for each class of images.
    This includes preparing the training data, estimating the optimal number of 
    concepts to be discovered (if not predefined), and finally training the model.
    
    Parameters:
    - class_names: List of class names to train on.
    - prediction_filter: Whether to filter the data based on classification confidence.
    - certainty_filter: Whether to filter the data based on concept certainty.
    - num_concepts: The number of concepts to discover. If None, it is estimated.
    """
    for class_name in class_names:
        results_path = os.path.join('results', 'training_results', class_name)
        clear_directory(results_path)
        data_path = os.path.join('results', 'training_data', class_name + '_train')
        clear_directory(data_path)

        data_list = prepare_data(
            class_name, data_path, prediction_filter, certainty_filter, num_concepts
        )
        
        training_dff(num_concepts, data_list, class_name)


def load_and_preprocess_images(data_list):
    """
    Loads and preprocesses images for inference, adjusting dimensions and color 
    channels as necessary.

    Parameters:
        - data_list (list of str): List of paths to the images to be processed.

    Returns:
        - ndarray: An array containing the processed images.
    """
    raw_images = [plt.imread(filename) for filename in data_list]
    for i, img in enumerate(raw_images):
        if len(img.shape) == 2:
            raw_images[i] = np.repeat(img[..., np.newaxis], 3, -1)
    return np.stack([imresize(img, 224, 224) for img in raw_images])


def run_inference_pipeline(data_path, class_name):
    """
    Executes the inference pipeline of concept discovery for a given class of images.
    It processes images from the specified data_path to generate and visualize
    concept heatmaps overlayed on the original images.
    
    Parameters:
    - data_path: Path to the directory containing images for inference.
    - class_name: The name of the class for which to run inference.
    """
    results_path = os.path.join('results', 'training_results', class_name)
    pre_file = os.path.join(results_path, class_name + '.txt')
    data_list = [
        os.path.join(data_path, filename) for filename in os.listdir(data_path)
    ] if os.path.isdir(data_path) else [data_path]

    orig_concept_heatmaps = inference_dff(data_list, pre_file)
    concept_heatmaps = orig_concept_heatmaps / \
        orig_concept_heatmaps.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    concept_heatmaps = concept_heatmaps.cpu().numpy()
    
    raw_images = load_and_preprocess_images(data_list)

    for v in range(raw_images.shape[0]):
        num_concepts = concept_heatmaps.shape[1]
        _cmap = matplotlib.colormaps['gist_rainbow']
        colors = [np.array(_cmap(i)[:3]) for i in np.linspace(0, 1, num_concepts)]

        img = raw_images[v]
        fig, axes = plt.subplots(1, num_concepts)
        for k in range(num_concepts):
            layer = np.ones((*img.shape[:2], 4))
            for c in range(3):
                layer[:, :, c] *= colors[k][c]
            mask = concept_heatmaps[v, k]
            layer[:, :, 3] = mask
            axes[k].imshow(img)
            axes[k].imshow(layer, alpha=0.75)
            axes[k].axis('off')

        plt.tight_layout(pad=0, w_pad=0.1, h_pad=0)
        plt.show()
