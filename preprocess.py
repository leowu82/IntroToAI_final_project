import os
import random
from collections import defaultdict

# Define the paths
input_folder = 'data/UTKFace'
output_folder_train = 'data/train'
output_folder_val = 'data/validation'

# Create the output folders if they do not exist
os.makedirs(output_folder_train, exist_ok=True)
os.makedirs(output_folder_val, exist_ok=True)

# Dictionary to store filenames by age
age_dict = defaultdict(list)
male_files = []
female_files = []

# Process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        # Extract the age from the filename
        parts = filename.split('_')
        age = parts[0]
        gender = parts[1]
        # Add the filename to the list of the corresponding age
        age_dict[age].append(filename)
        if gender == '0':  # Male
            male_files.append(filename)
        elif gender == '1':  # Female
            female_files.append(filename)

# Split the data and write the filenames to the corresponding age text files
for age, files in age_dict.items():
    random.shuffle(files)  # Shuffle the list of files for each age to ensure random split
    split_index = int(0.8 * len(files))  # 80% for training, 20% for validation
    train_files = files[:split_index]
    val_files = files[split_index:]

    # Write training files
    with open(os.path.join(output_folder_train, f'{age}.txt'), 'w') as f:
        for file in train_files:
            f.write(file + '\n')

    # Write validation files
    with open(os.path.join(output_folder_val, f'{age}.txt'), 'w') as f:
        for file in val_files:
            f.write(file + '\n')

# Shuffle the lists to ensure random split
random.shuffle(male_files)
random.shuffle(female_files)

# Split the data
split_index_male = int(0.8 * len(male_files))  # 80% for training, 20% for validation
split_index_female = int(0.8 * len(female_files))

train_male_files, val_male_files = male_files[:split_index_male], male_files[split_index_male:]
train_female_files, val_female_files = female_files[:split_index_female], female_files[split_index_female:]

# Write male training files
with open(os.path.join(output_folder_train, 'male.txt'), 'w') as f:
    for file in train_male_files:
        f.write(file + '\n')

# Write female training files
with open(os.path.join(output_folder_train, 'female.txt'), 'w') as f:
    for file in train_female_files:
        f.write(file + '\n')

# Write male validation files
with open(os.path.join(output_folder_val, 'male.txt'), 'w') as f:
    for file in val_male_files:
        f.write(file + '\n')

# Write female validation files
with open(os.path.join(output_folder_val, 'female.txt'), 'w') as f:
    for file in val_female_files:
        f.write(file + '\n')


print("Train and validation text files have been generated in the 'output/train' and 'output/val' folders.")
