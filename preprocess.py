import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
data_dir = 'data/img_align_celeba'
attr_file = 'data/list_attr_celeba.txt'
train_dir = 'data/train'
val_dir = 'data/validation'

# Load the attributes file
attributes = pd.read_csv(attr_file, delim_whitespace=True, header=1)

# Create labels for gender and age (young)
attributes['gender'] = attributes['Male'].apply(lambda x: 'male' if x == 1 else 'female')

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(attributes, test_size=0.2, random_state=42)

def create_directory_structure(base_dir, categories):
    for category in categories:
        os.makedirs(os.path.join(base_dir, category), exist_ok=True)

# Create directory structures
create_directory_structure(train_dir, ['male', 'female'])
create_directory_structure(val_dir, ['male', 'female'])

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

# Write filenames to text files for training and validation sets
write_filenames_to_txt(train_df, train_dir)
write_filenames_to_txt(val_df, val_dir)
