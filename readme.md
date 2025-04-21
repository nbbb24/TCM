# Tongue Coating Classification

This project is focused on classifying tongue coating images using Vision Transformer (ViT), ResNet, and CNN models, with additional capabilities for GPT-based analysis and a web interface. The dataset consists of images labeled with different tongue coating types, and the models are trained to predict these labels.

## Data Source

The dataset used in this project is sourced from the [YOLO Tongue Coating Dataset](https://github.com/jiangjiaqing/yolo_tongue_coating/tree/main/yolo_data/images). The dataset is organized into the following structure:

```
data/
├── train/    # Training images
└── test/     # Testing images
```

## Project Structure

```
.
├── CV/                      # Main classification scripts and utilities
│   ├── vit.py              # Vision Transformer model implementation
│   ├── vit0.py             # Original Vision Transformer implementation
│   ├── resnet.py           # ResNet model implementation
│   ├── resnet0.py          # Original ResNet implementation
│   ├── cnn.py              # CNN model implementation
│   ├── cnn0.py             # Original CNN implementation
│   ├── gpt.py              # GPT-based analysis implementation
│   ├── gpt_test_result.txt # Results from GPT analysis
│   ├── frontend_TCM.py     # Web interface implementation
│   ├── evaluate.py         # Model evaluation script
│   ├── utils.py            # Utility functions
│   ├── normal_class.py     # Normal class classification
│   ├── dataset_description.py # Dataset information
│   ├── extract_label.py    # Utility for extracting paths and labels
│   ├── find_label.py       # Script for finding and processing labels
│   ├── combine_image.py    # Script for combining images
│   ├── class_label         # Mapping of class labels to integers
│   ├── tinyllama/          # TinyLlama model implementation
│   ├── results/            # Directory for training results and plots
│   └── __pycache__/        # Python cache files
├── data/                   # Main data directory
├── model_weights/          # Directory for trained model weights
├── requirements.txt        # Python dependencies
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
   - Run normal_class.py to download and set up the normal class data:
     ```
     python CV/normal_class.py
     ```

## Usage

### Model Training and Evaluation
You can train and evaluate different models:

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

4. **Model Evaluation**:
   ```bash
   python CV/evaluate.py
   ```

### Web Interface
To run the web interface:
```bash
python CV/frontend_TCM.py
```

### GPT Analysis
To perform GPT-based analysis:
```bash
python CV/gpt.py
```

## Features

1. **Traditional Models**:
   - Vision Transformer (ViT)
   - ResNet
   - CNN

2. **Advanced Analysis**:
   - GPT-based analysis
   - TinyLlama integration
   - Web interface for easy interaction

3. **Evaluation Tools**:
   - Comprehensive model evaluation
   - Performance metrics
   - Visualization tools

## Model Selection

The project provides multiple approaches to tongue coating classification:

1. **Traditional Models**:
   - Vision Transformer (ViT): A transformer-based architecture
   - ResNet: A deep convolutional neural network with residual connections
   - CNN: A convolutional neural network

2. **Advanced Analysis**:
   - GPT-based analysis for detailed insights
   - TinyLlama for efficient processing
   - Web interface for user-friendly interaction

## Improving Model Performance

To improve model accuracy, consider:

- Experimenting with different data augmentation techniques
- Adjusting hyperparameters such as learning rate and batch size
- Using learning rate schedulers
- Implementing cross-validation
- Trying different model architectures or configurations
- Leveraging GPT analysis for additional insights

## Results

- Training and validation metrics are saved in the `results/` directory
- Model weights are saved in the `model_weights/` directory
- Performance metrics are printed during training and evaluation
- GPT analysis results are available in `gpt_test_result.txt`

## License

This project is licensed under the MIT License.
