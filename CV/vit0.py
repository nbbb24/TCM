import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, fold=None):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    class_accuracies = {}
    
    # For tracking per-class performance
    all_classes = train_loader.dataset.dataset.label_to_idx if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset.label_to_idx
    num_classes = len(all_classes)
    
    # Create a training log dictionary for this fold
    training_log = {
        "fold": fold,
        "epochs": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs} {'(Fold '+str(fold+1)+')' if fold is not None else ''}")
        print("-" * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize confusion matrix for this epoch
        confusion_matrix = torch.zeros(num_classes, num_classes)
        
        # Use tqdm for progress bar during training
        train_bar = tqdm(train_loader, desc="Training")
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
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'acc': f"{100 * correct / total:.2f}%"
            })
        
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Calculate per-class accuracy for training
        per_class_acc_train = {}
        for i in range(num_classes):
            if confusion_matrix[i].sum().item() > 0:
                per_class_acc_train[i] = confusion_matrix[i, i].item() / confusion_matrix[i].sum().item() * 100
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Reset confusion matrix for validation
        confusion_matrix = torch.zeros(num_classes, num_classes)
        
        # Use tqdm for progress bar during validation
        val_bar = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Update confusion matrix
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                
                # Update progress bar
                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'acc': f"{100 * val_correct / val_total:.2f}%"
                })
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Calculate per-class accuracy for validation
        per_class_acc_val = {}
        for i in range(num_classes):
            if confusion_matrix[i].sum().item() > 0:
                per_class_acc_val[i] = confusion_matrix[i, i].item() / confusion_matrix[i].sum().item() * 100
        
        # Convert class indices to names for easier reading
        idx_to_label = {v: k for k, v in all_classes.items()}
        per_class_acc_val_named = {idx_to_label[i]: acc for i, acc in per_class_acc_val.items()}
        
        epoch_time = time.time() - epoch_start
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Print per-class validation accuracy
        print("\nPer-class validation accuracy:")
        for class_name, acc in per_class_acc_val_named.items():
            print(f"  {class_name}: {acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model_weights/resnet.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            
            # Save per-class accuracies for the best model
            class_accuracies = per_class_acc_val_named
        
        # Add data to training log
        training_log["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "per_class_acc": per_class_acc_val_named,
            "duration_seconds": epoch_time
        })
    
    total_time = time.time() - start_time
    training_log["total_duration"] = total_time
    training_log["best_val_acc"] = best_val_acc
    training_log["best_class_accuracies"] = class_accuracies
    
    print(f'\nTraining completed in {total_time/60:.2f} minutes')
    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    # Save training log
    os.makedirs('results', exist_ok=True)
    log_filename = f'results/resnet_training_log_fold_{fold}.json' if fold is not None else 'results/resnet_training_log.json'
    with open(log_filename, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    return best_val_acc, train_losses, val_losses, train_accs, val_accs, class_accuracies

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    # For tracking per-class performance
    all_classes = test_loader.dataset.label_to_idx
    num_classes = len(all_classes)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    test_start = time.time()
    print("\nEvaluating model on test set...")
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing")
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            # Update progress bar
            test_bar.set_postfix({'acc': f"{100 * correct / total:.2f}%"})
    
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    per_class_acc = {}
    idx_to_label = {v: k for k, v in all_classes.items()}
    
    for i in range(num_classes):
        if confusion_matrix[i].sum().item() > 0:
            per_class_acc[idx_to_label[i]] = confusion_matrix[i, i].item() / confusion_matrix[i].sum().item() * 100
    
    test_time = time.time() - test_start
    
    print(f'\nTest evaluation completed in {test_time:.2f}s')
    print(f'Overall Test Accuracy: {accuracy:.2f}%')
    print("\nPer-class test accuracy:")
    for class_name, acc in per_class_acc.items():
        print(f"  {class_name}: {acc:.2f}%")
    
    # Save test results
    test_results = {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_matrix.tolist(),
        "duration_seconds": test_time
    }
    
    with open('results/resnet_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [idx_to_label[i] for i in range(num_classes)], rotation=45)
    plt.yticks(tick_marks, [idx_to_label[i] for i in range(num_classes)])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('results/confusion_matrix.png')
    
    return accuracy, per_class_acc

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create necessary directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    os.makedirs('model_weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = TongueDataset('data/train', transform=transform)
    test_dataset = TongueDataset('data/test', transform=test_transform)
    
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Cross-validation setup
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    fold_class_accuracies = []
    
    # Save training configuration
    config = {
        "model": "ResNet50",
        "lr": 0.0001,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "num_epochs": 10,
        "k_folds": k_folds,
        "image_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "augmentation": {
            "horizontal_flip": True,
            "rotation": 10,
            "color_jitter": True
        }
    }
    
    with open('results/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Get number of classes based on unique labels
    num_classes = len(train_dataset.label_to_idx)
    print(f"\nTraining ResNet50 model for {num_classes} classes with {k_folds}-fold cross-validation")
    
    # Cross-validation training
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'\n{"="*20} Fold {fold + 1}/{k_folds} {"="*20}')
        
        # Create data loaders for this fold
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_subsampler, num_workers=4)
        val_loader = DataLoader(train_dataset, batch_size=16, sampler=val_subsampler, num_workers=4)
        
        print(f"Training set: {len(train_idx)} samples, Validation set: {len(val_idx)} samples")
        
        # Initialize ResNet50 with pre-trained weights
        print("Initializing ResNet50 model with pre-trained weights...")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace the final fully connected layer for our number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
        
        # Train the model
        best_val_acc, train_losses, val_losses, train_accs, val_accs, class_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            num_epochs=10, device=device, fold=fold
        )
        
        fold_results.append(best_val_acc)
        fold_class_accuracies.append(class_accs)
        
        # Plot and save losses for this fold
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold+1} Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Training Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Fold {fold+1} Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/fold_{fold+1}_plots.png')
        plt.close()
    
    cv_time = time.time() - start_time
    
    # Print cross-validation results
    print('\n' + '='*50)
    print('Cross-validation results:')
    for i, acc in enumerate(fold_results):
        print(f'Fold {i+1}: {acc:.2f}%')
    print(f'Average accuracy: {np.mean(fold_results):.2f}% ± {np.std(fold_results):.2f}%')
    print(f'Cross-validation completed in {cv_time/60:.2f} minutes')
    
    # Save cross-validation results
    cv_results = {
        "fold_accuracies": {f"fold_{i+1}": acc for i, acc in enumerate(fold_results)},
        "mean_accuracy": float(np.mean(fold_results)),
        "std_accuracy": float(np.std(fold_results)),
        "duration_minutes": cv_time/60,
        "fold_class_accuracies": {f"fold_{i+1}": accs for i, accs in enumerate(fold_class_accuracies)}
    }
    
    with open('results/cross_validation_results.json', 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    # Load best model and evaluate on test set
    print('\n' + '='*50)
    print("Loading best model for final test evaluation...")
    model.load_state_dict(torch.load('model_weights/resnet.pth'))
    test_accuracy, per_class_acc = evaluate_model(model, test_loader, device)
    print(f'\nTest set accuracy: {test_accuracy:.2f}%')
    
    # Final timing information
    total_time = time.time() - start_time
    print(f'Total execution time: {total_time/60:.2f} minutes')

if __name__ == '__main__':
    main()