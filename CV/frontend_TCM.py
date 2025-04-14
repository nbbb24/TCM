import streamlit as st
import os
from PIL import Image
import tempfile
from gpt import predict_image, get_tcm_advice, friendly_names, class_labels

# Set page config
st.set_page_config(
    page_title="TCM Tongue Diagnosis",
    page_icon="👅",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Traditional Chinese Medicine Tongue Diagnosis")
st.markdown("""
This AI-powered application analyzes tongue images to provide Traditional Chinese Medicine (TCM) insights.
Upload a clear image of your tongue to receive:
- A diagnosis of your tongue condition
- Detailed TCM advice
- Personalized recommendations
""")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Your Tongue Image")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Tongue Image", use_container_width=True)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            
            # Add predict button
            if st.button("Analyze Tongue and Generate TCM Advice", type="primary"):
                try:
                    # Make prediction
                    with st.spinner("Analyzing tongue condition..."):
                        try:
                            predicted_label, confidence, all_probs = predict_image(tmp_file.name)
                            # Get the original class name from the label
                            original_class = next((k for k, v in class_labels.items() if v == predicted_label), None)
                            if original_class is None:
                                raise ValueError(f"Invalid predicted label: {predicted_label}")
                            # Get the friendly name
                            predicted_condition = friendly_names[original_class]
                            
                            # Display results in a nice box
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.success(f"**Diagnosis:** {predicted_condition}")
                            st.info(f"**Confidence:** {confidence:.2%}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Display detailed probabilities
                            st.subheader("Detailed Analysis")
                            for i, prob in enumerate(all_probs):
                                # Get the original class name for this index
                                original_class = next((k for k, v in class_labels.items() if v == i), None)
                                if original_class is None:
                                    continue  # Skip if no matching class found
                                # Get the friendly name
                                condition_name = friendly_names[original_class]
                                st.progress(prob, text=f"{condition_name}: {prob:.2%}")
                            
                            # Generate and display TCM advice
                            with st.spinner("Generating TCM advice..."):
                                tcm_advice = get_tcm_advice(predicted_condition)
                            
                            st.markdown('<div class="result-box">', unsafe_allow_html=True)
                            st.subheader("TCM Advice")
                            st.markdown(tcm_advice)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        except ValueError as ve:
                            st.markdown('<div class="error-box">', unsafe_allow_html=True)
                            st.error("**Validation Error**")
                            st.write(str(ve))
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.info("Please ensure you have uploaded a clear image of your tongue and try again.")
                            
                        except FileNotFoundError as fe:
                            st.markdown('<div class="error-box">', unsafe_allow_html=True)
                            st.error("**File Error**")
                            st.write(str(fe))
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.info("Please ensure the model file exists and try again.")
                            
                        except Exception as e:
                            st.markdown('<div class="error-box">', unsafe_allow_html=True)
                            st.error("**Analysis Error**")
                            st.write(f"Error type: {type(e).__name__}")
                            st.write(f"Error message: {str(e)}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.info("An unexpected error occurred. Please try again or contact support.")
                
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.error("**System Error**")
                    st.write(f"Error type: {type(e).__name__}")
                    st.write(f"Error message: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file.name)
                    except Exception as e:
                        st.warning(f"Warning: Could not clean up temporary file: {str(e)}")
    else:
        st.info("Please upload a tongue image to begin analysis.")

with col2:
    st.subheader("Tips for Best Results")
    st.markdown("""
    For accurate diagnosis, please follow these guidelines when taking the photo:
    
    - 📸 Use good lighting (natural light is best)
    - 👅 Extend your tongue fully
    - 🚫 Avoid eating or drinking 30 minutes before
    - 🎯 Focus on the tongue (avoid showing teeth)
    - 📱 Take the photo from a consistent angle
    
    **Common Tongue Conditions:**
    - Red tongue with thick, greasy coating
    - White tongue with thick, greasy coating
    - Black tongue coating
    - Geographic tongue (map-like coating)
    - Normal healthy tongue
    - Purple tongue coating
    - Red tongue with yellow, thick, greasy coating
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: gray; font-size: small;'>
        *Note: This is an AI-assisted tool and should not replace professional medical advice.
        Always consult with a qualified TCM practitioner for proper diagnosis and treatment.*
    </p>
</div>
""", unsafe_allow_html=True) 