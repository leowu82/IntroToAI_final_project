import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained models
gender_model = tf.keras.models.load_model('face_classification_gender_model.h5')
age_model = tf.keras.models.load_model('face_classification_age_model.h5')

# Compile the models to ensure metrics are available
# gender_model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
gender_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
age_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Function to preprocess input photo
def preprocess_image(image_path, target_size=(200,200)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to classify input photo for gender and predict age
def classify_and_predict_age(image_path):
    img1 = preprocess_image(image_path,(200,200))
    # Gender prediction
    gender_prediction = gender_model.predict(img1)
    gender_label = "Female" if gender_prediction[0][0] >= 0.5 else "Male"
    
    img2 = preprocess_image(image_path,(200,200))
    # Age prediction
    age_prediction = age_model.predict(img2)
    age_label = int(age_prediction[0][0])
    
    return gender_label, age_label

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
            
            # Classify the photo and predict age
            gender_label, age_label = classify_and_predict_age(file_path)
            
            # Print the predicted labels
            print(f"File: {file}, Predicted Gender: {gender_label}, Predicted Age: {age_label}")

# Example usage
input_folder_path = 'crop_input'
classify_photos_in_folder(input_folder_path)
