import os
import shutil

# Define the source directories
train_dir = 'yolo_data/images/train'
val_dir = 'yolo_data/images/val'

# Define the destination directory
combined_dir = 'yolo_data/images/all'

# Create the destination directory if it doesn't exist
os.makedirs(combined_dir, exist_ok=True)

# Function to copy images from a source directory to the destination directory
def copy_images(source_dir, dest_dir):
    for filename in os.listdir(source_dir):
        # Construct full file path
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        
        # Copy the file
        if os.path.isfile(source_file):
            shutil.copy(source_file, dest_file)

# Copy images from both train and val directories
copy_images(train_dir, combined_dir)
copy_images(val_dir, combined_dir)

print(f"Images have been combined into {combined_dir}")