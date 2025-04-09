import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from timm import create_model

# Define the class labels mapping
class_labels = {
    'The_white_tongue_is_thick_and_greasy': 0, 
    'red_tongue_yellow_fur_thick_greasy_fur': 1, 
    'The_red_tongue_is_thick_and_greasy': 2, 
    'black_tongue_coating': 3, 
    'map_tongue_coating_': 4, 
    'purple_tongue_coating': 5
}

class TongueDataset(Dataset):
    def __init__(self, label_file, image_dir='yolo_data/images/all', transform=None):
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.image_dir = image_dir  # Base directory for images
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    label_str = parts[1]
                    # Convert the string label to an integer using the mapping
                    label = class_labels.get(label_str)
                    if label is not None:
                        self.image_files.append(img_path)
                        self.labels.append(label)
                    else:
                        print(f"Warning: Label '{label_str}' not found in class_labels.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    train_losses = []  # Track training losses
    val_losses = []    # Track validation losses
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
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
        
        train_acc = 100 * correct / total
        train_losses.append(running_loss / len(train_loader))  # Record training loss
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_running_loss / len(val_loader))  # Record validation loss
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model_weights/best_model.pth')
    
    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('results/loss_plot.png')  # Save the plot as an image file
    plt.show()

def test_model(model, test_loader, device):
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data transforms
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(10),
       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
    
    # Create datasets with corrected label file paths
    train_dataset = TongueDataset('yolo_data/label/train.txt', transform=transform)
    val_dataset = TongueDataset('yolo_data/label/val.txt', transform=transform)
    test_dataset = TongueDataset('yolo_data/label/test.txt', transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Create model
    model = create_model('vit_base_patch16_224', pretrained=True, num_classes=len(class_labels))
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    
    # Test the model
    test_model(model, test_loader, device)

if __name__ == '__main__':
    main()