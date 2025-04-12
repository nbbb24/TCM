# Tongue Coating Classification

This project is focused on classifying tongue coating images using Vision Transformer (ViT), ResNet, and CNN models. The dataset consists of images labeled with different tongue coating types, and the models are trained to predict these labels.

## Data Source

The dataset used in this project is sourced from the [YOLO Tongue Coating Dataset](https://github.com/jiangjiaqing/yolo_tongue_coating/tree/main/yolo_data/images). The dataset is organized into the following structure:

```
data/
├── train/    # Training images
└── val/     # Testing images
```

## Project Structure

```
.
├── CV/                      # Main classification scripts and utilities
│   ├── vit.py              # Vision Transformer model implementation
│   ├── resnet.py           # ResNet model implementation
│   ├── cnn.py              # CNN model implementation
│   ├── extract_label.py    # Utility for extracting paths and labels from image filenames
│   ├── find_label.py       # Script for finding and processing labels
│   ├── combine_image.py    # Script for combining images
│   ├── class_label         # Mapping of class labels to integers
│   ├── requirements.txt    # Python dependencies
│   ├── results/            # Directory for training results and plots
├── data/                   # Main data directory
├── model_weights/          # Directory for trained model weights
└── readme.md              # Project documentation
```

## Environment Setup

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

4. **Setup Data**: 
   - Navigate to the project directory
   - Download the dataset from the provided source and rename it as data
   - Ensure the data is properly organized in the `data/` directory
   - Rename val folder as test

## Usage

### Model Training
You can train either the Vision Transformer, ResNet, or CNN model:

1. **Vision Transformer (ViT)**:
   ```bash
   python CV/vit.py
   ```

2. **ResNet**:
   ```bash
   python CV/resnet.py
   ```

3. **CNN**:
   ```bash
   python CV/cnn.py
   ```

All scripts will:
- Train the model
- Save the best model weights in the `model_weights/` directory
- Generate training and validation plots in the `results/` directory
- Evaluate the model on the test set

## Model Selection

The project provides three different approaches to tongue coating classification:

1. **Vision Transformer (ViT)**: A transformer-based architecture that processes images as sequences of patches.
2. **ResNet**: A deep convolutional neural network with residual connections.
3. **CNN**: A convolutional neural network.

You can experiment with all models to determine which performs better for your specific use case.

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
