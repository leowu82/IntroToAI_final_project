import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained models
gender_model = tf.keras.models.load_model('face_classification_gender_model.h5')
age_model = tf.keras.models.load_model('face_classification_age_model.h5')

# Function to preprocess input photo
def preprocess_image(image_path, target_size=(200,200)):
  """Preprocesses an image for model input."""
  img = image.load_img(image_path, target_size=target_size)
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array / 255.0  # Normalize the image
  return img_array

# Function to classify input photo for gender and predict age
def classify_and_predict_age(image_path):
  """Classifies gender and predicts age for a given image."""
  img = preprocess_image(image_path)
  # Gender prediction
  gender_prediction = gender_model.predict(img)
  gender_label = "Female" if gender_prediction[0][0] >= 0.5 else "Male"
  
  # Age prediction
  #age_prediction = age_model.predict(img)
  #age_label = int(age_prediction[0][0])
  
  return gender_label#, age_label
 
def calculate_gender_accuracy(data_folder_path, label_folder_path):
  """Calculate the accuracy of gender model"""
  with open(os.path.join(label_folder_path,"male.txt"), 'r') as file:
    male_filenames = file.read().splitlines()

  with open(os.path.join(label_folder_path,"female.txt"), 'r') as file:
    female_filenames = file.read().splitlines()
  
  wrong_count = 0 
  for male_filename in male_filenames:
    file_path = os.path.join(data_folder_path, male_filename)
    gender_label = classify_and_predict_age(file_path)
    if gender_label != "Male":
      wrong_count = wrong_count + 1
  
  for female_filename in female_filenames:
    file_path = os.path.join(data_folder_path, female_filename)
    gender_label = classify_and_predict_age(file_path)
    if gender_label != "Female":
      wrong_count = wrong_count + 1  

  total = len(male_filenames) + len(female_filenames)
  accuracy = 1 - wrong_count/total
  print(f"Data Folder: {label_folder_path}, Model Accuracy: {accuracy}")
  return accuracy

# Example usage
input_folder_path = 'crop_input'
data_folder_path = "data/UTKFace"
label_folder_path1 = "data/train"
label_folder_path2 = "data/validation"
accuracy1 = calculate_gender_accuracy(data_folder_path, label_folder_path1)
accuracy2 = calculate_gender_accuracy(data_folder_path, label_folder_path2)
print(f"Data Folder: {label_folder_path1}, Model Accuracy: {accuracy1}")
print(f"Data Folder: {label_folder_path2}, Model Accuracy: {accuracy2}")

