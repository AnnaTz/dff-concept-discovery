import skimage.transform
import os
from PIL import Image, ImageOps
import numpy as np
import math
from kneed import KneeLocator
import shutil


def find_shape(x, y):
    """
    Determine the trend and curvature of a dataset.
    
    Parameters:
    - x (array): The x-coordinates of the data points.
    - y (array): The y-coordinates of the data points.
    
    Returns:
    - tuple: A tuple containing the trend ('increasing' or 'decreasing') and the 
    curvature ('concave' or 'convex').
    """
    p = np.polyfit(x, y, deg=1)
    x1, x2 = int(len(x) * 0.2), int(len(x) * 0.8)
    q = np.mean(y[x1:x2]) - np.mean(x[x1:x2] * p[0] + p[1])
    if p[0] > 0 and q > 0:
        return 'increasing', 'concave'
    if p[0] > 0 and q <= 0:
        return 'increasing', 'convex'
    if p[0] <= 0 and q > 0:
        return 'decreasing', 'concave'
    return 'decreasing', 'convex'


def find_crit_point(x, y):
    """
    Find a critical point in the dataset using the KneeLocator library.
    
    Parameters:
    - x (array): The x-coordinates of the data points.
    - y (array): The y-coordinates of the data points.
    
    Returns:
    - The critical point (knee or elbow) in the dataset.
    """
    x = list(x)
    direction, curve = find_shape(x, y)
    kneedle_train = KneeLocator(x, y, S=1.0, curve=curve, direction=direction)
    return kneedle_train.knee if curve == 'concave' else kneedle_train.elbow


def concat_images(image_paths, size, shape=None, border_width=3):
    """
    Concatenate multiple images into a single image with a specified layout.
    
    Parameters:
    - image_paths (list): List of image file paths to concatenate.
    - size (tuple): The target size for each image part (width, height).
    - shape (tuple): The grid's shape (rows, cols). If None, defaults to a single row.
    - border_width (int): The width of the border around each image part.
    
    Returns:
    - Image: The concatenated image.
    """
    width, height = size
    width += 2 * border_width
    height += 2 * border_width
    images = [
        ImageOps.pad(Image.open(image), size, color='grey') 
        for image in image_paths
    ]

    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size)
    
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            if idx < len(image_paths):
                image.paste(
                    ImageOps.expand(images[idx], border=border_width, fill='black'), 
                    offset,
                )
    
    return image


def make_grid(source_path, dest_path, grid_name, part_size):
    """
    Create a grid of images from a source directory and save it to a destination.
    
    Parameters:
    - source_path (str): Path to the source directory containing images.
    - dest_path (str): Path to the destination directory to save the grid image.
    - grid_name (str): Name of the grid image file.
    - part_size (tuple): Size of each part in the grid (width, height).
    """
    image_paths = [
        os.path.join(source_path, f) 
        for f in os.listdir(source_path) 
        if f.lower().endswith(('png', 'jpg', 'jpeg'))
    ]
    images_count = len(image_paths)

    grid_width = math.ceil(math.sqrt(images_count))
    grid_height = math.ceil(images_count / grid_width)

    image = concat_images(image_paths, part_size, (grid_height, grid_width))
    os.makedirs(dest_path, exist_ok=True)
    image.save(os.path.join(dest_path, grid_name))


def imresize(img, height=None, width=None):
    """
    Resize an image to a specified width and/or height.
    
    Parameters:
    - img (numpy.ndarray): The image to resize.
    - height (int): The target height. If None, the aspect ratio is preserved.
    - width (int): The target width. If None, the aspect ratio is preserved.
    
    Returns:
    - numpy.ndarray: The resized image.
    """
    if height is not None and width is not None:
        ny, nx = height, width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny, nx = img.shape[0], img.shape[1]
    return skimage.transform.resize(img, (int(ny), int(nx)), mode='constant')


def clear_directory(path):
    """Safely removes a directory and its contents if it exists."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception as e:
        print(f"Error removing directory {path}: {e}")
        