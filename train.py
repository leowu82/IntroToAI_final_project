import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

# Paths
train_dir = 'data/img_align_celeba'
val_dir = 'data/validation'
male_filenames_txt = 'data/train/male.txt'
female_filenames_txt = 'data/train/female.txt'

# Image dimensions
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)

# Number of epochs
epochs = 3

# Batch size
batch_size = 32

# Load filenames for male and female images
with open(male_filenames_txt, 'r') as file:
    male_filenames = file.read().splitlines()

with open(female_filenames_txt, 'r') as file:
    female_filenames = file.read().splitlines()

# Create data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_male_images = [os.path.join(train_dir, filename) for filename in male_filenames]
train_female_images = [os.path.join(train_dir, filename) for filename in female_filenames]
train_images = train_male_images + train_female_images
train_labels = [0] * len(male_filenames) + [1] * len(female_filenames)

train_data = list(zip(train_images, train_labels))
np.random.shuffle(train_data)

def custom_generator(data, batch_size):
    i = 0
    while True:
        batch_images = []
        batch_labels = []
        for _ in range(batch_size):
            if i == len(data):
                i = 0
                np.random.shuffle(data)
            image_path, label = data[i]
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
            image = tf.keras.preprocessing.image.img_to_array(image)
            batch_images.append(image)
            batch_labels.append(label)
            i += 1
        yield (np.array(batch_images), np.array(batch_labels))

train_generator = custom_generator(train_data, batch_size)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

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
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          steps_per_epoch=len(train_data) // batch_size,
          epochs=epochs,
          validation_data=val_generator,
          validation_steps=len(val_generator))

# Save the trained model
model.save('face_classification_model.h5')
