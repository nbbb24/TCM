import os
import random
from collections import defaultdict

def rename_images(image_dir):
    for image_name in os.listdir(image_dir):
        # Replace spaces with underscores in the filename
        new_image_name = image_name.replace(' ', '_')
        if new_image_name != image_name:
            os.rename(
                os.path.join(image_dir, image_name),
                os.path.join(image_dir, new_image_name)
            )

def extract_all_labels(image_dir):
    all_data = defaultdict(list)
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        label = os.path.splitext(image_name)[0]
        label = '_'.join(label.split('_')[:-1])
        all_data[label].append(image_path)
    return all_data

def split_and_write_data(all_data, label_dir, train_ratio=0.7, val_ratio=0.15):
    train_data, val_data, test_data = [], [], []
    
    for label, images in all_data.items():
        random.shuffle(images)
        total_count = len(images)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        
        train_data.extend((image, label) for image in images[:train_count])
        val_data.extend((image, label) for image in images[train_count:train_count + val_count])
        test_data.extend((image, label) for image in images[train_count + val_count:])
    
    write_data_to_file(train_data, os.path.join(label_dir, 'train.txt'))
    write_data_to_file(val_data, os.path.join(label_dir, 'val.txt'))
    write_data_to_file(test_data, os.path.join(label_dir, 'test.txt'))

def write_data_to_file(data, file_path):
    with open(file_path, 'w') as file:
        for image_path, label in data:
            file.write(f"{image_path} {label}\n")

# Define paths
image_dir = 'yolo_data/images/all'
label_dir = 'yolo_data/label'

# Ensure the label directory exists
os.makedirs(label_dir, exist_ok=True)

# Rename images to replace spaces with underscores
rename_images(image_dir)

# Extract all labels and paths
all_data = extract_all_labels(image_dir)

# Split the data and write to files
split_and_write_data(all_data, label_dir)