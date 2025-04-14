import torch
from torchvision import transforms
from PIL import Image
from timm import create_model
import os
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import platform

# Check if running on Apple Silicon
IS_MAC_M1 = platform.processor() == 'arm' and platform.system() == 'Darwin'

# Class labels mapping
class_labels =  {'The red tongue is thick and greasy': 0, 'The white tongue is thick and greasy': 1, 'black tongue coating': 2, 'map tongue coating_': 3, 'normal_class': 4, 'purple tongue coating': 5, 'red tongue yellow fur thick greasy fur': 6}
# Mapping from original class names to human-friendly names
friendly_names = {
    'The red tongue is thick and greasy': "Red tongue with thick, greasy coating",
    'The white tongue is thick and greasy': "White tongue with thick, greasy coating",
    'black tongue coating': "Black tongue coating",
    'map tongue coating_': "Geographic tongue (map-like coating)",
    'normal_class': "Normal healthy tongue",
    'purple tongue coating': "Purple tongue coating",
    'red tongue yellow fur thick greasy fur': "Red tongue with yellow, thick, greasy coating"
}

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

Please provide a comprehensive TCM consultation for the following tongue condition, divided into three sections. Each section must include exactly 3 bullet points, written in clear and professional language for patients.
Keep each bullet concise and focused on practical, actionable advice suitable for patients and do not repeat the headings in your answer.

Format your output as follows:

### TCM Diagnosis
- describing possible internal imbalances or organ dysfunctions
- listing common accompanying symptoms or patterns

### Dietary Recommendations
- foods or substances to avoid
- any dietary habits or patterns to follow

### Lifestyle Guidance
- beneficial daily routines or habits
- activities or environments to avoid
"""
            },
            {
                "role": "user",
                "content": f"""The type of tongue condition is {tongue_condition}"""}]
        
        # Set device based on platform
        if IS_MAC_M1:
            device = "mps"  # Metal Performance Shaders for Apple Silicon
            print("Using Apple Silicon GPU (MPS)")
        else:
            device = 0 if torch.cuda.is_available() else -1
            print(f"Using {'GPU' if device == 0 else 'CPU'}")
        
        # Initialize pipeline with device configuration
        pipe = pipeline(
            "text-generation", 
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=device,
            torch_dtype=torch.float16 if device != -1 else torch.float32,
            model_kwargs={"load_in_8bit": True} if IS_MAC_M1 else {}  # Use 8-bit quantization on M1
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

def predict_image(image_path, model_path='model_weights/vit.pth', num_classes=7):
    """
    Predict the label of a single image using a trained Vision Transformer model.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the saved model weights
        num_classes (int): Number of classes in the model
    
    Returns:
        tuple: (predicted_label, confidence_score, all_probabilities)
    """
    try:
        # Set device based on platform
        if IS_MAC_M1:
            device = torch.device('mps')  # Metal Performance Shaders for Apple Silicon
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
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
            
            # Convert to Python types
            predicted_label = predicted.item()
            confidence_score = confidence.item()
            all_probabilities = probabilities[0].tolist()
            
            # Verify the predicted label is valid
            if predicted_label not in class_labels.values():
                raise ValueError(f"Invalid predicted label: {predicted_label}")
            
            return predicted_label, confidence_score, all_probabilities
            
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    image_path = "data/test/red tongue yellow fur thick greasy fur_169.jpg"
    try:
        print("Image path:", image_path)
        predicted_label, confidence, all_probs = predict_image(image_path)
        predicted_condition_raw = label_to_name[predicted_label]
        predicted_condition = friendly_names[predicted_condition_raw]
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