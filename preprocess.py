import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
data_dir = 'data/img_align_celeba'
attr_file = 'data/list_attr_celeba.txt'
train_dir = 'data/train'
val_dir = 'data/validation'

# Num of photos
num_photos = 20000

# Load the attributes file
attributes = pd.read_csv(attr_file, sep='\s+', header=1)

# Constrain the dataset to the num_photos
attributes = attributes.iloc[:num_photos]

# Create labels
attributes['gender'] = attributes['Male'].apply(lambda x: 'male' if x == 1 else 'female')

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(attributes, test_size=0.2, random_state=42)

def create_directory_structure(base_dir):
    os.makedirs(base_dir, exist_ok=True)

# Function to write filenames to a text file for each gender
def write_filenames_to_txt(df, dest_dir):
    male_filenames = df[df['gender'] == 'male'].index.tolist()
    female_filenames = df[df['gender'] == 'female'].index.tolist()

    with open(os.path.join(dest_dir, 'male.txt'), 'w') as male_file:
        for filename in male_filenames:
            male_file.write(filename + '\n')

    with open(os.path.join(dest_dir, 'female.txt'), 'w') as female_file:
        for filename in female_filenames:
            female_file.write(filename + '\n')

# Create directory structures
create_directory_structure(train_dir)
create_directory_structure(val_dir)

# Write filenames to text files for training and validation sets
write_filenames_to_txt(train_df, train_dir)
write_filenames_to_txt(val_df, val_dir)

print("Training and validation text files have been created.")
