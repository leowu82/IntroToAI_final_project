import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths
train_dir = 'data/UTKFace'
val_dir = 'data/UTKFace'
train_male_filenames_txt = 'data/train/male.txt'
train_female_filenames_txt = 'data/train/female.txt'
val_male_filenames_txt = 'data/validation/male.txt'
val_female_filenames_txt = 'data/validation/female.txt'

# Image dimensions
img_width, img_height = 200, 200
input_shape = (img_width, img_height, 3) #RGB
 
# Number of epochs
epochs = 3

# Batch size
batch_size = 32

# Load filenames for male and female images for training
with open(train_male_filenames_txt, 'r') as file:
    train_male_filenames = file.read().splitlines()

with open(train_female_filenames_txt, 'r') as file:
    train_female_filenames = file.read().splitlines()

# Load filenames for male and female images for validation
with open(val_male_filenames_txt, 'r') as file:
    val_male_filenames = file.read().splitlines()

with open(val_female_filenames_txt, 'r') as file:
    val_female_filenames = file.read().splitlines()

# Create data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_male_images = [os.path.join(train_dir, filename) for filename in train_male_filenames]
train_female_images = [os.path.join(train_dir, filename) for filename in train_female_filenames]
train_images = train_male_images + train_female_images
train_labels = [0] * len(train_male_filenames) + [1] * len(train_female_filenames)

val_male_images = [os.path.join(val_dir, filename) for filename in val_male_filenames]
val_female_images = [os.path.join(val_dir, filename) for filename in val_female_filenames]
val_images = val_male_images + val_female_images
val_labels = [0] * len(val_male_filenames) + [1] * len(val_female_filenames)

train_data = list(zip(train_images, train_labels))
val_data = list(zip(val_images, val_labels))

def custom_generator(data, batch_size):
    num_samples = len(data)
    while True:
        indices = np.random.choice(num_samples, batch_size, replace=False)
        batch_images = []
        batch_labels = []
        for i in indices:
            image_path, label = data[i]
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0  # Normalize
            batch_images.append(image)
            batch_labels.append(label)
        yield (np.array(batch_images), np.array(batch_labels))

# Define steps per epoch
steps_per_epoch = len(train_data) // batch_size
validation_steps = len(val_data) // batch_size

train_generator = custom_generator(train_data, batch_size)
val_generator = custom_generator(val_data, batch_size)

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Combine base model and custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
# model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=0.003, clipvalue=1.0), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=val_generator,
          validation_steps=validation_steps)

# Save the trained model
model.save('face_classification_gender_model.h5')
