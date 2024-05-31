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
race_dict = defaultdict(list)
gender_dict = defaultdict(list)

Gender = {
    '0': 'male',
    '1': 'female'
}

# Race = {
#     '0': 'White',
#     '1': 'Black',
#     '2': 'Asian',
#     '3': 'Indian',
#     '4': 'Other'
# }

# Process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        # Extract the age, gender, and race from the filename
        parts = filename.split('_')
        age = parts[0]
        gender = parts[1]
        # race = parts[2]
        
        # Add the filename to the list of the corresponding age, gender, and race
        age_dict[age].append(filename)
        gender_dict[Gender.get(gender)].append(filename)
        # race_dict[Race.get(race)].append(filename)

# Split the data and write the filenames to the corresponding text files
def split_and_write_files(data_dict, output_folder):
    for category, files in data_dict.items():
        random.shuffle(files)  # Shuffle the list of files for each category to ensure random split
        split_index = int(0.8 * len(files))  # 80% for training, 20% for validation
        train_files = files[:split_index]
        val_files = files[split_index:]

        # Write training files
        with open(os.path.join(output_folder_train, f'{category}.txt'), 'w') as f:
            for file in train_files:
                f.write(file + '\n')

        # Write validation files
        with open(os.path.join(output_folder_val, f'{category}.txt'), 'w') as f:
            for file in val_files:
                f.write(file + '\n')

split_and_write_files(age_dict, output_folder_train)
split_and_write_files(age_dict, output_folder_val)

split_and_write_files(gender_dict, output_folder_train)
split_and_write_files(gender_dict, output_folder_val)

# split_and_write_files(race_dict, output_folder_train)
# split_and_write_files(race_dict, output_folder_val)

print("Train and validation text files have been generated in the 'output/train' and 'output/val' folders.")
