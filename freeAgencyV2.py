import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Load and sort the dataset
df = pd.read_csv('sorted_filtered_batting_season_summary.csv')
df_sorted = df.sort_values(by='Year')

# Determine the cutoff year
cutoff_year = df_sorted['Year'].quantile(0.8, interpolation='higher')

# Split the dataset
train_data = df_sorted[df_sorted['Year'] <= cutoff_year]
test_data = df_sorted[df_sorted['Year'] > cutoff_year]

# Preparing weights for the training data - later years have higher weights
weights = train_data['Year'] - train_data['Year'].min() + 1  # Ensure no weight is 0

# Features and target
features_train = train_data[['age', 'HR', 'H', '2B', '3B']]
target_train = train_data['total']
features_test = test_data[['age', 'HR', 'H', '2B', '3B']]
target_test = test_data['total']

# Train the model with weights
model = LinearRegression()
model.fit(features_train, target_train, sample_weight=weights)

# Predictions
predictions = model.predict(features_test)

# Evaluation
rmse = sqrt(mean_squared_error(target_test, predictions))
r_squared = r2_score(target_test, predictions)

# Predicted data into DataFrame
predictions_df = test_data.copy()
predictions_df['Predicted Hits'] = predictions
predictions_df = predictions_df[['Name', 'Year', 'age', 'HR', 'H', '2B', '3B', 'total', 'Predicted Hits']]

print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")
predictions_df = predictions_df[['Name', 'Year', 'age', 'HR', 'H', '2B', '3B', 'total', 'Predicted Hits']]

# Show the first few rows of the predictions DataFrame
predictions_df.to_csv("predictionsWeightedByAge.csv", index=False)