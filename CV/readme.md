# Tongue Coating Classification

This project is focused on classifying tongue coating images using a Vision Transformer (ViT) model. The dataset consists of images labeled with different tongue coating types, and the model is trained to predict these labels.

## Project Structure

- **CV/Classification/**: Contains the main classification scripts and data.
  - **vit.py**: The main script for training and testing the Vision Transformer model.
  - **extract_label.py**: Utility script for extracting paths and labels from image filenames that saved as train/test/val text.
  - **find_label.py**: Script for finding and processing labels.
  - **combine_image.py**: Script for combining images, if applicable.
  - **class_label**: Contains the mapping of class labels to integers.
  - **requirements.txt**: Lists the Python dependencies required for the project.
  - **model_weights/**: Directory where trained model weights are saved.
  - **results/**: Directory where results such as plots are saved.
  - **yolo_data/**: Contains the dataset.
    - **images/**: Directory containing all images.
      - **all/**: Subdirectory with all images used for training, validation, and testing.
    - **label/**: Contains text files with image paths and labels.
      - **train.txt**: Training dataset labels.
      - **val.txt**: Validation dataset labels.
      - **test.txt**: Test dataset labels.

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

## Setup


1. **Install Dependencies**: Ensure you have Python installed, then install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**: Ensure that your images are placed in the `yolo_data/images/all` directory and that labels are correctly extracted using `extract_label.py`.

3. **Train the Model**: Run the `vit.py` script to train the model. The script will save the best model weights.

4. **Evaluate the Model**: The script will also evaluate the model on the test set and print the accuracy.

## Usage

- **Training**: Modify the hyperparameters and training settings in `vit.py` as needed, then execute the script to start training.
- **Evaluation**: After training, the model can be evaluated using the test dataset to check its performance.

## Results

- Training and validation loss plots are saved in the `results/` directory.
- The best model weights are saved in the `model_weights/` directory.

## Improving Accuracy

- Experiment with different data augmentation techniques.
- Adjust hyperparameters such as learning rate and batch size.
- Consider using a learning rate scheduler for better convergence.

## License

This project is licensed under the MIT License.
