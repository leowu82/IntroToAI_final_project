import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
file_path = 'spotify_dataset/songs_normalize.csv'
df = pd.read_csv(file_path)

# # Check for null values
# df.info()
# print(df.isnull().sum())
# # => none 

# # Replace null values with the mean of their respective columns
# df.fillna(df.mean(), inplace=True)

# Drop unused rows
df = df[df['explicit'] != True]
df = df[df['year'] != 2020]

# Drop unused columns
df.drop(['explicit', 'genre'], axis=1, inplace=True)

# Replace 0 values in 'popularity' to mean
mean_excluding_zero = df.loc[df['popularity'] != 0, 'popularity'].mean()
df['popularity'].replace(0, mean_excluding_zero, inplace=True)

# Select columns to scale
columns_to_scale = [
    'popularity', 'duration_ms', 'acousticness', 'danceability', 
    'instrumentalness', 'energy', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo'
]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Scale the selected columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# sort df by 'year'
df.sort_values(by='year', inplace=True)

# Save the preprocessed data to a new CSV file
output_file_path = 'spotify_dataset/songs_preprocessed.csv'
df.to_csv(output_file_path, index=False)