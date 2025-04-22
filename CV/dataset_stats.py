import os
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def count_images_per_class(dataset_path):
    """Count the number of images per class in a dataset directory."""
    class_counts = defaultdict(int)
    total_images = 0
    
    # Get all image files in the directory
    for img_name in os.listdir(dataset_path):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            # Get label from image name (everything before the last underscore)
            label = '_'.join(img_name.split('_')[:-1])
            class_counts[label] += 1
            total_images += 1
    
    return dict(class_counts), total_images

def plot_class_distribution(train_counts, test_counts, output_file):
    """Plot the distribution of classes in both training and test sets."""
    # Get all unique classes
    all_classes = sorted(set(train_counts.keys()) | set(test_counts.keys()))
    
    # Prepare data for plotting
    train_values = [train_counts.get(cls, 0) for cls in all_classes]
    test_values = [test_counts.get(cls, 0) for cls in all_classes]
    
    # Create the plot
    x = np.arange(len(all_classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, train_values, width, label='Training Set')
    rects2 = ax.bar(x + width/2, test_values, width, label='Test Set')
    
    # Add labels and title
    ax.set_ylabel('Number of Images')
    ax.set_title('Distribution of Images per Class')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    # Define paths
    train_path = 'data/train'
    test_path = 'data/test'
    output_file = 'CV/results/dataset_distribution.png'
    
    # Create results directory if it doesn't exist
    os.makedirs('CV/results', exist_ok=True)
    
    # Count images in both datasets
    train_counts, train_total = count_images_per_class(train_path)
    test_counts, test_total = count_images_per_class(test_path)
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Total images in training set: {train_total}")
    print(f"Total images in test set: {test_total}")
    print(f"Total images: {train_total + test_total}")
    print("\nImages per class:")
    print("-" * 50)
    
    # Get all unique classes
    all_classes = sorted(set(train_counts.keys()) | set(test_counts.keys()))
    
    # Print counts for each class
    for cls in all_classes:
        train_count = train_counts.get(cls, 0)
        test_count = test_counts.get(cls, 0)
        total_count = train_count + test_count
        train_percent = (train_count / total_count * 100) if total_count > 0 else 0
        test_percent = (test_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\nClass: {cls}")
        print(f"  Training: {train_count} images ({train_percent:.1f}%)")
        print(f"  Test: {test_count} images ({test_percent:.1f}%)")
        print(f"  Total: {total_count} images")
    
    # Create and save the distribution plot
    plot_class_distribution(train_counts, test_counts, output_file)
    print(f"\nDistribution plot saved to: {output_file}")

if __name__ == '__main__':
    main() 