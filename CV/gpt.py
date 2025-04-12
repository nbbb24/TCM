import torch
from torchvision import transforms
from PIL import Image
from timm import create_model
import os
from transformers import pipeline

# Class labels mapping
class_labels = {'The red tongue is thick and greasy': 0, 'The white tongue is thick and greasy': 1, 'black tongue coating': 2, 'map tongue coating_': 3, 'purple tongue coating': 4, 'red tongue yellow fur thick greasy fur': 5}

# Reverse the mapping for easy lookup
label_to_name = {v: k for k, v in class_labels.items()}

def get_tcm_advice(tongue_condition):
    """
    Generate Traditional Chinese Medicine advice based on tongue coating diagnosis.
    
    Args:
        tongue_condition (str): The diagnosed tongue condition
    
    Returns:
        str: TCM advice and recommendations
    """
    # Initialize the pipeline
    pipe = pipeline("text-generation", model="Qwen/Qwen-7B-Chat", trust_remote_code=True)
    
    # Create a prompt for TCM advice
    prompt = f"""As a Traditional Chinese Medicine practitioner, provide detailed advice for a patient with the following tongue condition: {tongue_condition}.
    
Please include:
1. The TCM diagnosis and its meaning
2. Possible underlying health issues
3. Dietary recommendations
4. Lifestyle suggestions
5. Herbal medicine recommendations (if applicable)
6. Acupressure points to focus on

Format the response in a clear, professional manner suitable for a patient consultation."""

    # Create messages for the pipeline
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Generate response
    response = pipe(messages)
    return response[0]['generated_text']

def predict_image(image_path, model_path='model_weights/vit.pth', num_classes=6):
    """
    Predict the label of a single image using a trained Vision Transformer model.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the saved model weights
        num_classes (int): Number of classes in the model
    
    Returns:
        tuple: (predicted_label, confidence_score, all_probabilities)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")
    
    # Load the model
    model = create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        raise ValueError(f"Error loading model weights: {str(e)}")
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities[0].tolist()

# Example usage
if __name__ == "__main__":
    image_path = "data/test/red tongue yellow fur thick greasy fur_169.jpg"
    try:
        print("Image path:", image_path)
        predicted_label, confidence, all_probs = predict_image(image_path)
        predicted_condition = label_to_name[predicted_label]
        print(f"\nPredicted condition: {predicted_condition}")
        print(f"Confidence: {confidence:.4f}")
        
        print("\nProbabilities for each condition:")
        for i, prob in enumerate(all_probs):
            print(f"{label_to_name[i]}: {prob:.4f}")
        
        print("\nGenerating TCM advice...")
        tcm_advice = get_tcm_advice(predicted_condition)
        print("\nTCM Advice:")
        print(tcm_advice)
        
    except Exception as e:
        print(f"Error: {str(e)}")