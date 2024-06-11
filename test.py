import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from Music_recommend import spotify_api

current_year = 2024

# Load the trained models
gender_model = tf.keras.models.load_model('face_classification_gender_model.h5')
age_model = tf.keras.models.load_model('face_classification_age_model.h5')

# Function to preprocess input photo
def preprocess_image(image_path, target_size=(200,200)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to classify input photo for gender and predict age
def classify_and_predict_age(image_path):
    img = preprocess_image(image_path)
    # Gender prediction
    gender_prediction = gender_model.predict(img)
    gender_label = "Female" if gender_prediction[0][0] >= 0.5 else "Male"
    
    # Age prediction
    age_prediction = age_model.predict(img)
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
            year = current_year - age_label + min(age_label, 18)
            df_all_songs = spotify_api.fetch_desired_year_songs(year)
            
            # Print song names and singers
            if not df_all_songs.empty:
                # Randomly select 5 songs
                random_songs = df_all_songs.sample(n=3)
                for _, song in random_songs.iterrows():
                    print(f"Title: {song['title']}, Artist: {song['artist']}\nSong URL: {song['spotify_url']}")
            else:
                print(f"No songs found for year {year}")

# Example usage
input_folder_path = 'crop_input'
classify_photos_in_folder(input_folder_path)
