import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

def load_image(image_file):
    img = Image.open(image_file).convert('RGB')
    return img

def preprocess_image(image, target_dim=512):
    img = image.resize((target_dim, target_dim))
    img = tf.image.convert_image_dtype(np.array(img), tf.float32)
    img = img[tf.newaxis, ...]  # Add batch dimension
    return img

@st.cache(allow_output_mutation=True)
def load_style_transfer_model():
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
    return model

def perform_style_transfer(content_image, style_image, model):
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    return stylized_image

st.title('Neural Style Transfer with Streamlit')

content_file = st.sidebar.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])
style_file = st.sidebar.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])

if content_file and style_file:
    content_image = load_image(content_file)
    style_image = load_image(style_file)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Content Image")
        st.image(content_image, use_column_width=True)
    with col2:
        st.header("Style Image")
        st.image(style_image, use_column_width=True)

    if st.button('Transfer Style'):
        model = load_style_transfer_model()
        content_image_processed = preprocess_image(content_image)
        style_image_processed = preprocess_image(style_image)
        stylized_image = perform_style_transfer(content_image_processed, style_image_processed, model)
        
        # Convert the tensor to a NumPy array and clip values to ensure they are in the [0, 1] range
        stylized_image_np = stylized_image.numpy()
        stylized_image_np = np.squeeze(stylized_image_np)
        stylized_image_np = np.clip(stylized_image_np, 0, 1)
        
        # Convert NumPy array to PIL Image for resizing
        stylized_image_pil = Image.fromarray((stylized_image_np * 255).astype(np.uint8))
        
        # Resize the stylized image
        stylized_image_resized = stylized_image_pil.resize((400, 400))  # Adjust size as needed

        st.header("Stylized Image")
        st.image(stylized_image_resized)
