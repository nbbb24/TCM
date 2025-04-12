import os
from PIL import Image
from pathlib import Path

class ImageDataset:
    def __init__(self, image_dir):
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

def main():
    # Define paths
    train_dir = 'data/train'
    test_dir = 'data/test'
    
    print("=== Analyzing Train Dataset ===")
    train_dataset = ImageDataset(train_dir)
    
    print("\n=== Analyzing Test Dataset ===")
    test_dataset = ImageDataset(test_dir)
    
    # Print overall statistics
    total_images = len(train_dataset.image_files) + len(test_dataset.image_files)
    print(f"\nTotal images in dataset: {total_images}")
    print(f"Train images: {len(train_dataset.image_files)} ({len(train_dataset.image_files)/total_images*100:.1f}%)")
    print(f"Test images: {len(test_dataset.image_files)} ({len(test_dataset.image_files)/total_images*100:.1f}%)")

if __name__ == "__main__":
    main()