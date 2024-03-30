import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Load the dataset
df = pd.read_csv('sorted_filtered_batting_season_summary.csv')

# Sort the dataset by 'Year'
df_sorted = df.sort_values(by='Year')

# Determine the year to split the data: first 80% for training, last 20% for testing
cutoff_year = df_sorted['Year'].quantile(0.8, interpolation='higher')

# Split the dataset based on the cutoff year
train_data = df_sorted[df_sorted['Year'] <= cutoff_year]
test_data = df_sorted[df_sorted['Year'] > cutoff_year]

# Selecting features and target
features_train = train_data[['age', 'AB', 'H', '2B', '3B']]
features_test = test_data[['age', 'AB', 'H', '2B', '3B']]
target_train = train_data['total']
target_test = test_data['total']

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(features_train, target_train)

# Make predictions on the test set
predictions = model.predict(features_test)

# Evaluate the model
rmse = sqrt(mean_squared_error(target_test, predictions))
r_squared = r2_score(target_test, predictions)
print(f"Root Mean Squared Error (RMSE) for Test Set: {rmse}")
print(f"R-squared (Coefficient of Determination) for Test Set: {r_squared}")


# Create a new DataFrame with predictions
predictions_df = test_data.copy()
predictions_df['Predicted Hits'] = predictions

# Select relevant columns to display (including player names, actual and predicted hits)
predictions_df = predictions_df[['Name', 'Year', 'age', 'AB', 'H', '2B', '3B', 'total', 'Predicted Hits']]

# Show the first few rows of the predictions DataFrame
predictions_df.to_csv("predictions.csv", index=False)
