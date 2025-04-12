import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from timm import create_model
import numpy as np
from sklearn.model_selection import KFold
import shutil
from tqdm import tqdm
import requests
from urllib3.exceptions import IncompleteRead
import time

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    # Create results directory if it doesn't exist
    os.makedirs('CV/results/vit_plots', exist_ok=True)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Open log file
    log_file = open('CV/results/vit_log.txt', 'w')
    log_file.write('Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\n')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
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
            batch_count += 1
            
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
            for images, labels in val_loader:
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
            torch.save(model.state_dict(), 'model_weights/vit.pth')
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
    plt.savefig('CV/results/vit_plots/vit_training_curves.png')
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

def download_file_with_retry(url, destination, max_retries=3, chunk_size=8192):
    """
    Download a file with retry logic and progress bar.
    
    Args:
        url (str): URL of the file to download
        destination (str): Local path to save the file
        max_retries (int): Maximum number of retry attempts
        chunk_size (int): Size of chunks to download at a time
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)
            
            print(f"Successfully downloaded {destination}")
            return True
            
        except (requests.exceptions.RequestException, IncompleteRead) as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to download after {max_retries} attempts")
                return False

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create necessary directories
    os.makedirs('CV/data/train', exist_ok=True)
    os.makedirs('CV/data/test', exist_ok=True)
    os.makedirs('CV/model_weights', exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TongueDataset('CV/data/train', transform=transform)
    test_dataset = TongueDataset('CV/data/test', transform=transform)
    
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Cross-validation setup
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []
    
    # Create model with number of classes based on unique labels
    num_classes = len(train_dataset.label_to_idx)
    model = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001)
    
    # Cross-validation training
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'\nFold {fold + 1}/{k_folds}')
        
        # Create data loaders for this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_subsampler, num_workers=4)
        val_loader = DataLoader(train_dataset, batch_size=16, sampler=val_subsampler, num_workers=4)
        
        # Train the model
        best_val_acc, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            num_epochs=10, device=device
        )
        fold_results.append(best_val_acc)
    
    # Print cross-validation results
    print('\nCross-validation results:')
    for i, acc in enumerate(fold_results):
        print(f'Fold {i+1}: {acc:.2f}%')
    print(f'Average accuracy: {np.mean(fold_results):.2f}% ± {np.std(fold_results):.2f}%')
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('model_weights/vit.pth'))
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f'\nTest set accuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    main()