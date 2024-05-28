import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('face_classification_model.h5')

# Function to preprocess input photo
def preprocess_image(image_path, target_size=(178, 218)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to classify input photo
def classify_photo(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    # Assuming binary classification (e.g., gender), use threshold to determine label
    if prediction >= 0.5:
        label = "Female"
    else:
        label = "Male"
    return label

# Function to classify all photos in a folder
def classify_photos_in_folder(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through each file
    for file in files:
        # Check if the file is a JPG file
        if file.endswith(".jpg"):
            # Get the full path of the file
            file_path = os.path.join(folder_path, file)
            
            # Classify the photo
            predicted_label = classify_photo(file_path)
            
            # Print the predicted label
            print(f"File: {file}, Predicted label: {predicted_label}")

# Example usage
input_folder_path = 'resized_images'
classify_photos_in_folder(input_folder_path)
