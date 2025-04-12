import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import time
from datetime import datetime
import json
from tqdm import tqdm

class TongueDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.transform = transform
        self.image_files = []
        self.labels = []
        
        # Get all image files in the directory
        for img_name in os.listdir(image_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(image_dir, img_name)
                # Get label from image name (everything before the last underscore)
                label = '_'.join(img_name.split('_')[:-1])
                self.image_files.append(img_path)
                self.labels.append(label)
        
        # Create label to index mapping
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        # Convert string labels to indices
        self.labels = [self.label_to_idx[label] for label in self.labels]
        
        print(f"Found {len(self.image_files)} images with {len(unique_labels)} unique labels")
        print("Label mapping:", self.label_to_idx)
        
        # Count images per class
        class_counts = {}
        for label in self.labels:
            class_name = list(self.label_to_idx.keys())[list(self.label_to_idx.values()).index(label)]
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        print("Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images ({count/len(self.image_files)*100:.1f}%)")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_next_file_number(base_name, extension, directory):
    """Get the next available file number for the given base name and extension."""
    i = 1
    while True:
        filename = f"{base_name}_{i}{extension}" if i > 1 else f"{base_name}{extension}"
        if not os.path.exists(os.path.join(directory, filename)):
            return i
        i += 1

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name, hyperparameters):
    # Create results and logs directories if they don't exist
    os.makedirs(f'CV/results/{model_name}_plots', exist_ok=True)
    os.makedirs(f'CV/results/{model_name}_logs', exist_ok=True)
    
    # Get next available file numbers
    plot_num = get_next_file_number(f'{model_name}_training_curves', '.png', f'CV/results/{model_name}_plots')
    log_num = get_next_file_number(f'{model_name}_log', '.txt', f'CV/results/{model_name}_logs')
    
    # Create filenames
    plot_filename = f"{model_name}_training_curves_{plot_num}.png" if plot_num > 1 else f"{model_name}_training_curves.png"
    log_filename = f"{model_name}_log_{log_num}.txt" if log_num > 1 else f"{model_name}_log.txt"
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Open log file
    log_file = open(os.path.join(f'CV/results/{model_name}_logs', log_filename), 'w')
    
    # Write hyperparameters to log file
    log_file.write('Hyperparameters:\n')
    for key, value in hyperparameters.items():
        log_file.write(f'{key}: {value}\n')
    log_file.write('\nTraining Progress:\n')
    log_file.write('Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\n')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase with tqdm
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
        
        train_acc = 100 * correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_acc)
        
        # Validation phase with tqdm
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * val_correct / val_total:.2f}%"
                })
        
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_running_loss / len(val_loader))
        val_accuracies.append(val_acc)
        
        # Write to log file
        log_file.write(f'{epoch+1}\t{running_loss/len(train_loader):.4f}\t{train_acc:.2f}\t'
                      f'{val_running_loss/len(val_loader):.4f}\t{val_acc:.2f}\n')
        
        # Print epoch summary
        print('\n' + '='*50)
        print(f'Epoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_running_loss/len(val_loader):.4f}, Validation Accuracy: {val_acc:.2f}%')
        print('='*50 + '\n')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'model_weights/{model_name}.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
    
    # Print final training summary
    print('\nTraining Completed!')
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    print(f'Final Training Accuracy: {train_accuracies[-1]:.2f}%')
    print(f'Final Validation Accuracy: {val_accuracies[-1]:.2f}%')
    
    # Write final summary to log file
    log_file.write('\nFinal Summary:\n')
    log_file.write(f'Best Validation Accuracy: {best_val_acc:.2f}%\n')
    log_file.write(f'Final Training Accuracy: {train_accuracies[-1]:.2f}%\n')
    log_file.write(f'Final Validation Accuracy: {val_accuracies[-1]:.2f}%\n')
    log_file.close()
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(f'CV/results/{model_name}_plots', plot_filename))
    plt.close()
    
    return best_val_acc, train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    test_bar = tqdm(test_loader, desc='Testing')
    with torch.no_grad():
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            test_bar.set_postfix({'acc': f"{100 * correct / total:.2f}%"})
    
    accuracy = 100 * correct / total
    return accuracy

def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_default_hyperparameters():
    return {
        'learning_rate': 0.00001,
        'batch_size': 16,
        'num_epochs': 10,
        'k_folds': 5,
        'image_size': 224,
        'optimizer': 'AdamW',
        'loss_function': 'CrossEntropyLoss',
        'data_augmentation': True
    }

def setup_directories():
    # Create necessary directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    os.makedirs('model_weights', exist_ok=True)
    os.makedirs('CV/results', exist_ok=True) 