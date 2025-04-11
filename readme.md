# Tongue Coating Classification

This project is focused on classifying tongue coating images using both Vision Transformer (ViT) and ResNet models. The dataset consists of images labeled with different tongue coating types, and the models are trained to predict these labels.

## Project Structure

- **CV/**: Contains the main classification scripts and data.
  - **vit.py**: Script for training and testing the Vision Transformer model.
  - **resnet.py**: Script for training and testing the ResNet model.
  - **extract_label.py**: Utility script for extracting paths and labels from image filenames.
  - **find_label.py**: Script for finding and processing labels.
  - **combine_image.py**: Script for combining images, if applicable.
  - **class_label**: Contains the mapping of class labels to integers.
  - **requirements.txt**: Lists the Python dependencies required for the project.
  - **model_weights/**: Directory where trained model weights are saved.
  - **results/**: Directory where results such as plots are saved.
  - **data/**: Contains the dataset.
    - **images/**: Directory containing all images.
    - **label/**: Contains text files with image paths and labels.

## Environment Setup

To create a conda environment for this project, follow these steps:

1. **Create a Conda Environment**: Use the following command to create a new conda environment named `tcm` with Python 3.10:
   ```
   conda create -n tcm python==3.10
   ```

2. **Activate the Environment**: Activate the newly created environment:
   ```
   conda activate tcm
   ```

3. **Install Dependencies**: Once the environment is activated, install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation
1. Place your images in the appropriate directory structure
2. Use `extract_label.py` to process and extract labels from your images
3. Ensure your data is properly organized in the `data/` directory

### Model Training
You can train either the Vision Transformer or ResNet model:

1. **Vision Transformer (ViT)**:
   ```bash
   python CV/vit.py
   ```

2. **ResNet**:
   ```bash
   python CV/resnet.py
   ```

Both scripts will:
- Train the model
- Save the best model weights in the `model_weights/` directory
- Generate training and validation plots in the `results/` directory
- Evaluate the model on the test set

## Model Selection

The project provides two different approaches to tongue coating classification:

1. **Vision Transformer (ViT)**: A transformer-based architecture that processes images as sequences of patches.
2. **ResNet**: A deep convolutional neural network with residual connections.

You can experiment with both models to determine which performs better for your specific use case.

## Improving Model Performance

To improve model accuracy, consider:

- Experimenting with different data augmentation techniques
- Adjusting hyperparameters such as learning rate and batch size
- Using learning rate schedulers
- Implementing cross-validation
- Trying different model architectures or configurations

## Results

- Training and validation metrics are saved in the `results/` directory
- Model weights are saved in the `model_weights/` directory
- Performance metrics are printed during training and evaluation

## License

This project is licensed under the MIT License.
