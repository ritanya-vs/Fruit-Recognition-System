import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

# Define the path where ESP32 will save the image
IMAGE_PATH = "C:/Users/Ritanya/Downloads/photo.jpg"
class_names = ['Apple', 'Banana']

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="C:/Users/Ritanya/Downloads/cnn_fruit_classification.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path, image_size):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image_size, image_size))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image_path):
    # Preprocess the image
    input_shape = input_details[0]['shape'][1]  # Assumes a square input shape
    image = preprocess_image(image_path, input_shape)

    # Set the model input
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output prediction
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    
    return predicted_class, predictions

# Streamlit Interface
st.title("ESP32-CAM Image Prediction")

# Continuously check if the image exists
while True:
    if os.path.exists(IMAGE_PATH):
        # Load the image and display it in Streamlit
        image = Image.open(IMAGE_PATH)
        st.image(image, caption='Last Captured Image', use_column_width=True)
        
        # Run prediction
        label, predictions = predict(IMAGE_PATH)
        if label is not None:
            st.write(f"Prediction: {label}") 
        
        # Remove the image after prediction (optional)
        os.remove(IMAGE_PATH)
        
        # Update the Streamlit page
        st.experimental_rerun()

    # Wait for a short interval before checking again
    time.sleep(1)
