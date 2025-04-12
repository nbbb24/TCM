import torch
from torchvision import transforms
from PIL import Image
from timm import create_model
import os
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    try:
        print("Creating prompt...")
        # Create chat messages with system and user roles
        messages = [{"role": "system",
                "content": """You are an expert Traditional Chinese Medicine (TCM) practitioner with extensive knowledge of tongue diagnosis and treatment.
                Please provide a comprehensive TCM consultation that includes:

1. TCM DIAGNOSIS
- Explain the meaning of this tongue condition in TCM terms
- Describe the underlying imbalances it indicates

2. DIETARY RECOMMENDATIONS
- Specific foods to include
- Foods to avoid
- Any dietary patterns to follow

3. LIFESTYLE GUIDANCE
- Daily habits to adopt
- Activities to avoid
- Stress management techniques

Please provide your response in a clear, professional format suitable for a patient consultation. Use no more than 3 sentences for each section. Focus on practical, actionable advice."""
            },
            {
                "role": "user",
                "content": f"""The type of tongue condition is {tongue_condition}"""}]
        
        # Set device
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using {'GPU' if device == 0 else 'CPU'}")
        
        # Initialize pipeline with device configuration
        pipe = pipeline(
            "text-generation", 
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=device,
            torch_dtype=torch.float16 if device == 0 else torch.float32
        )
        
        # Apply chat template and generate response
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(
            prompt,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        answer = outputs[0]["generated_text"]
        
        if not answer:
            print("Warning: Empty response received")
            return "I apologize, but I'm having trouble generating a response at the moment. Please try again."
        
        # Extract only the assistant's response
        assistant_start = answer.find("<|assistant|>")
        if assistant_start != -1:
            assistant_content = answer[assistant_start + len("<|assistant|>"):].strip()
            # Remove any remaining tags
            assistant_content = assistant_content.replace("<|system|>", "").replace("<|user|>", "").replace("</s>", "")
            
            # Save to file
            os.makedirs("CV/tinyllama", exist_ok=True)
            base_filename = "CV/tinyllama/tinyllama_output"
            counter = 1
            filename = f"{base_filename}.txt"
            
            while os.path.exists(filename):
                filename = f"{base_filename}_{counter}.txt"
                counter += 1
                
            with open(filename, "w", encoding="utf-8") as f:
                f.write(assistant_content)
            
            return assistant_content.strip()
            
        return answer
        
    except Exception as e:
        print(f"Error generating TCM advice: {str(e)}")
        return f"An error occurred while generating TCM advice: {str(e)}"

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
        # print(f"Confidence: {confidence:.4f}")
        
        # print("\nProbabilities for each condition:")
        # for i, prob in enumerate(all_probs):
        #     print(f"{label_to_name[i]}: {prob:.4f}")
        
        print("\nGenerating TCM advice...")
        tcm_advice = get_tcm_advice(predicted_condition)
        print("\nTCM Advice:")
        print(tcm_advice)
        
    except Exception as e:
        print(f"Error: {str(e)}")