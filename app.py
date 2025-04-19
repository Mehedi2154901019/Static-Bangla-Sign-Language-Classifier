import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
model = load_model("new50.keras")  #99.48% accuracy

# Define correct input size for Xception
IMG_SIZE = (300, 300)

# Class labels from 0 to 47
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'D', 'Dh', 'NG', 'R', 'T', 'Th', 
               'a', 'aa', 'b', 'bh', 'bisorgo', 'c', 'ch', 'dd', 'ddh', 'e', 'g', 'gh', 'h', 'i', 'j', 
               'jh', 'k', 'kh', 'l', 'm', 'n', 'nng', 'o', 'p', 'ph', 'rr', 's', 'space', 'tt', 'tth', 
               'u', 'y']

# Streamlit UI
st.title("Bangla Sign Language Classifier")
st.write("Upload an image of a Bangla sign language gesture to predict its class.")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display image
    img = image.load_img(uploaded_image, target_size=IMG_SIZE)
    st.image(img, caption="Uploaded Image", width=300)
    
    # Preprocess image for model
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0  # Normalize to [-1, 1]
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100  # Confidence percentage
    
    # Show result
    st.success(f"Predicted Class: **{predicted_class}** with {confidence:.2f}% confidence")
