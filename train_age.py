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
train_folder = 'data/train'
val_folder = 'data/validation'

# Image dimensions
img_width, img_height = 200, 200
input_shape = (img_width, img_height, 3)

# Number of epochs
epochs = 5

# Batch size
batch_size = 32

# Load filenames and ages for training
train_images = []
train_labels = []
for age_file in os.listdir(train_folder):
    try:
        age = int(age_file.split('.')[0])
        with open(os.path.join(train_folder, age_file), 'r') as file:
            filenames = file.read().splitlines()
            for filename in filenames:
                train_images.append(os.path.join(train_dir, filename))
                train_labels.append(age)
    except ValueError:
        continue  # Skip files that do not represent ages

# Load filenames and ages for validation
val_images = []
val_labels = []
for age_file in os.listdir(val_folder):
    try:
        age = int(age_file.split('.')[0])
        with open(os.path.join(val_folder, age_file), 'r') as file:
            filenames = file.read().splitlines()
            for filename in filenames:
                val_images.append(os.path.join(val_dir, filename))
                val_labels.append(age)
    except ValueError:
        continue  # Skip files that do not represent ages

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
predictions = Dense(1, activation='linear')(x)  # Linear activation for regression

# Combine base model and custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.002, clipvalue=1.0), loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(train_generator,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=val_generator,
          validation_steps=validation_steps)

# Save the trained model
model.save('face_classification_age_model.h5')
