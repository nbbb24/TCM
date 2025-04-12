import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import time
from datetime import datetime
import json
from tqdm import tqdm
from utils import (
    TongueDataset,
    train_model,
    evaluate_model,
    get_default_transform,
    get_default_hyperparameters,
    setup_directories
)

class ResNetModel:
    def __init__(self, num_classes, device):
        self.device = device
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(device)
        
    def get_model(self):
        return self.model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup directories
    setup_directories()
    
    # Get default hyperparameters
    hyperparameters = get_default_hyperparameters()
    
    # Data transforms
    transform = get_default_transform()
    
    # Create datasets
    train_dataset = TongueDataset('data/train', transform=transform)
    test_dataset = TongueDataset('data/test', transform=transform)
    
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], 
                           shuffle=False, num_workers=4)
    
    # Cross-validation setup
    k_folds = hyperparameters['k_folds']
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []
    
    # Create model with number of classes based on unique labels
    num_classes = len(train_dataset.label_to_idx)
    resnet_model = ResNetModel(num_classes, device)
    model = resnet_model.get_model()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'])
    
    # Cross-validation training
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'\nFold {fold + 1}/{k_folds}')
        
        # Create data loaders for this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], 
                                sampler=train_subsampler, num_workers=4)
        val_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], 
                              sampler=val_subsampler, num_workers=4)
        
        # Train the model
        best_val_acc, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            num_epochs=hyperparameters['num_epochs'], device=device,
            model_name='resnet', hyperparameters=hyperparameters
        )
        fold_results.append(best_val_acc)
    
    # Print cross-validation results
    print('\nCross-validation results:')
    for i, acc in enumerate(fold_results):
        print(f'Fold {i+1}: {acc:.2f}%')
    print(f'Average accuracy: {np.mean(fold_results):.2f}% ± {np.std(fold_results):.2f}%')
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('model_weights/resnet.pth'))
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f'\nTest set accuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    main()
