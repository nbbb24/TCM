import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from timm import create_model
from torchvision.models import resnet50, ResNet50_Weights
from cnn import SimpleCNN
from vit import TongueDataset
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    test_bar = tqdm(test_loader, desc='Testing')
    with torch.no_grad():
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    # Calculate overall accuracy
    accuracy = 100 * np.sum(np.diag(cm)) / np.sum(cm)
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return metrics

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n\n")
            
            f.write("Confusion Matrix:\n")
            cm = np.array(metrics['confusion_matrix'])
            f.write(np.array2string(cm, separator=', '))
            f.write("\n\n")
            
            f.write("Classification Report:\n")
            report = metrics['classification_report']
            for label, scores in report.items():
                if isinstance(scores, dict):
                    f.write(f"Class {label}:\n")
                    for metric, value in scores.items():
                        f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset and loader
    test_dataset = TongueDataset('data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Get number of classes
    num_classes = len(test_dataset.label_to_idx)
    
    # Initialize models
    models = {
        'CNN': SimpleCNN(num_classes),
        'ResNet': resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        'ViT': create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    }
    
    # Modify ResNet's final layer
    models['ResNet'].fc = nn.Linear(models['ResNet'].fc.in_features, num_classes)
    
    # Dictionary to store results
    all_results = {}
    
    # Load weights and evaluate each model
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Load model weights
        model_path = f'model_weights/{model_name.lower()}.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
            
            # Evaluate model
            metrics = evaluate_model(model, test_loader, device)
            all_results[model_name] = metrics
            print(f"{model_name} Test Accuracy: {metrics['accuracy']:.2f}%")
        else:
            print(f"Model weights not found at {model_path}")
    
    # Save results to file
    output_file = 'CV/results/evaluation.txt'
    save_results(all_results, output_file)
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main() 