import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Get the file path
file_path = os.path.join(os.path.dirname(__file__), 'movies.csv')

# Load the dataset
data = pd.read_csv(file_path, encoding='latin1')  # Adjust encoding as needed

# Check for NaN values
print("NaN values in Rating:", data['Rating'].isna().sum())
print("NaN values in Year:", data['Year'].isna().sum())
print("NaN values in Duration:", data['Duration'].isna().sum())
print("NaN values in Votes:", data['Votes'].isna().sum())

# Data cleaning
data['Year'] = data['Year'].str.extract(r'(\d+)').astype(float)  # Extract numeric values
data['Duration'] = data['Duration'].str.extract(r'(\d+)').astype(float)  # Extract duration
data['Votes'] = data['Votes'].str.replace(',', '').str.extract(r'(\d+)').astype(float)  # Remove commas

# Impute missing values for Year, Duration, and Votes
imputer = SimpleImputer(strategy='mean')
data[['Year', 'Duration', 'Votes']] = imputer.fit_transform(data[['Year', 'Duration', 'Votes']])

# Check for NaN values in Rating after cleaning
print("NaN values in Rating after imputing Year, Duration, and Votes:", data['Rating'].isna().sum())

# Optionally, drop rows with NaN values in 'Rating'
data = data.dropna(subset=['Rating'])

# Data types after cleaning
print("Data types after cleaning:")
print(data.dtypes)

# Define features and target variable
X = data[['Year', 'Duration', 'Votes']]
y = data['Rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on the test set
test_predictions = model.predict(X_test)
print("Predictions on test set:", test_predictions)

# Prepare predictions for the entire dataset
new_movies_df = data.copy()  # Use the cleaned data for predictions

# Select features for prediction
X_new = new_movies_df[['Year', 'Duration', 'Votes']]

# Make predictions
predictions = model.predict(X_new)

# Add predictions to the DataFrame
new_movies_df['Predicted Rating'] = predictions

# Output results
print(new_movies_df[['Name', 'Predicted Rating']])

# Optionally save to a new CSV
# new_movies_df.to_csv('predicted_movie_ratings.csv', index=False)  # Uncomment to save
