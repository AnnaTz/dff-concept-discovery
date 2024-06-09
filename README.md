
This project builds upon the methodologies discussed in "Deep Feature Factorization for Concept Discovery" by Edo Collins et al., ECCV 2018. The paper introduces an approach to understanding what deep learning models learn by extracting and visualizing high-level concepts from image data. The core idea revolves around using Non-negative Matrix Factorization (NMF) on the features extracted by a Convolutional Neural Network (CNN) classifier (specifically VGG-19). This effectively decomposes a set of images into semantic components (or 'concepts') that provide insights into the features the CNN has learned to focus on. 

### Key Features
- **Interpretable Deep Learning**: Implements Deep Feature Factorization (DFF), a method to make deep learning models more transparent by breaking down what the models learn from the data.
- **Training and Inference Pipelines**: Beyond replicating the DFF method, this project introduces custom training and inference pipelines for extracting and visualizing concepts.  
- **Training Dataset Optimization**: Fine-grains the training dataset to enhance the reliability and quality of concept discovery based on a novel metric of discovery confidence.
- **Number of Concepts Estimation**: Uses this metric to automatically estimate an optimal number of concepts that the model can discover in the training dataset.
- **Concept Visualization**: Utilizes advanced image processing techniques to visualize the heatmaps, extract significant regions, and contour them on images, making the learned features tangible and understandable.

### Project Structure
- `demo.ipynb`: An IPython notebook that showcases the use of the training and inference pipelines along with visualization of their results.
- `concept_discovery/`: A directory containing the Python modules for concept discovery.
  - `data.py`: Handles data loading, preprocessing, and fine-graining.
  - `dff.py`: Contains the logic for Deep Feature Factorization, including feature extraction, heatmap preparation, and concept confidence estimation.
  - `nmf.py`: Implements Non-negative Matrix Factorization.
  - `pipelines.py`: Defines the training and inference pipelines for concept discovery.
  - `utils.py`: Provides utility functions for image manipulation, directory management, and visualization.

### Getting Started
1. **Source Paper**: It is highly recommended to [read the paper](https://ivrlwww.epfl.ch/ecollins/deep_feature_factorization/) this project is based upon.
2. **Installation**: Ensure Python 3.8+ is installed. Then, use `pip install -r requirements.txt` to install the necessary dependencies.
3. **Run the Demo**: To discover concepts of an ImageNet class, add a directory of images named after the class to the `imagenet_data` directory. Then launch `demo.ipynb` with Jupyter Notebook to run the training and inference pipelines and visualize the discovered concepts.
