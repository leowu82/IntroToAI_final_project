# train classifier model to analyze feature importance of trending songs

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

file_path = 'spotify_dataset/songs_preprocessed.csv'
df = pd.read_csv(file_path)

# add feature: Like
df['Like'] = None

# can customize features that affect 'Like'
for index, row in df.iterrows():
    if row['valence'] >= 0.5 and row['loudness'] <= 0.9 and row['energy'] <= 0.9 and row['popularity'] >= 0.3:
        df.at[index, 'Like'] = 1
    else:
        df.at[index, 'Like'] = 0

# Shuffle the data to remove any biases
data_shuffled = df.sample(frac=1, random_state=42)

# equally distribute data with respect to column: 'year'
equal_distribute_col = 'year'
train_data, test_data = train_test_split(df, test_size=0.1, stratify=df[equal_distribute_col], random_state=42)

# drop unused columns
train_data = train_data.drop(['artist', 'song'], axis=1)


# ----------------------------------------
# divide dataset into training and testing
x = train_data.drop("Like", axis=1)
y = train_data["Like"].astype(int)

# Perform stratified train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Initialize the Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42, max_features=6, min_samples_split=30, min_samples_leaf=4)

# Train the model
random_forest.fit(x_train, y_train)

# Make predictions
predictions = random_forest.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# You can also print classification report for detailed evaluation
print("Classification Report:")
print(classification_report(y_test, predictions))

# ----------------------------------------
# Define the parameter grid to search over
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

# Create a RandomForestClassifier instance
rf = RandomForestClassifier(random_state=42)

# Setup the RandomizedSearchCV instance
random_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=0, n_jobs=-1)

# Fit the RandomizedSearchCV instance to the data
random_search.fit(x, y)

# Print the best parameters and the corresponding score
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_}")

# Retrieve the best estimator
best_rf = random_search.best_estimator_

# Initialize the Random Forest classifier
best_rf = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=10)

# Train the model
best_rf.fit(x_train, y_train)

# Make predictions
predictions = best_rf.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# You can also print classification report for detailed evaluation
print("Classification Report:")
print(classification_report(y_test, predictions))

feature_importances = pd.DataFrame({
    'Feature': x.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importances