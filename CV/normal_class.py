import os
import requests
from tqdm import tqdm
from PIL import Image
import io

# Create directories if they don't exist
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

# Base URL for raw GitHub content
base_url = "https://raw.githubusercontent.com/jiangjiaqing/yolo_tongue_coating/main/data/TongeImageDataset/img/"

# Download and split the first 320 images
for i in tqdm(range(1, 321)):
    filename = f"{i}.png"
    url = base_url + filename
    
    # Determine if this image goes to train or test
    if i <= 220:
        save_path = os.path.join('data/train', f"normal_class_{i}.png")
    else:
        save_path = os.path.join('data/test', f"normal_class_{i}.png")
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Save the PNG image directly
            with open(save_path, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")

print("Download completed!")
print(f"Training images: {len(os.listdir('data/train'))}")
print(f"Test images: {len(os.listdir('data/test'))}")