import torch
import torch.nn as nn
import torch.optim as optim
from utils import *

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # 224/2/2 = 56
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup directories and get default parameters
    setup_directories()
    hyperparameters = get_default_hyperparameters()
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
    model = SimpleCNN(num_classes)
    model = model.to(device)
    
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
            model_name='cnn', hyperparameters=hyperparameters
        )
        fold_results.append(best_val_acc)
    
    # Print cross-validation results
    print('\nCross-validation results:')
    for i, acc in enumerate(fold_results):
        print(f'Fold {i+1}: {acc:.2f}%')
    print(f'Average accuracy: {np.mean(fold_results):.2f}% ± {np.std(fold_results):.2f}%')
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('model_weights/cnn.pth'))
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f'\nTest set accuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    main()
