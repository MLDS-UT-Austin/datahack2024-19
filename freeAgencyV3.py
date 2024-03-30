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
train_data = df_sorted[df_sorted['Year'] < cutoff_year]
test_data = df_sorted[df_sorted['Year'] >= cutoff_year]

# Prepare year weights - later years have higher weights
year_weights = train_data['Year'] - train_data['Year'].min() + 1  # Ensure no weight is 0
min_weight = year_weights.min()


if min_weight <= 0:
    year_weights += (1 - min_weight) 

# Prepare age weights (simplified model for demonstration)
avg_hits_by_age = df.groupby('age')['total'].mean()
peak_performance_age = avg_hits_by_age.idxmax()
age_weights = train_data['age'].apply(lambda x: max(1, abs(peak_performance_age - abs(x - peak_performance_age))))

# Normalize and combine year and age weights
year_weights_normalized = abs(year_weights - year_weights.mean()) / year_weights.std()
age_weights_normalized = abs(age_weights - age_weights.mean()) / age_weights.std()
combined_weights = year_weights_normalized * age_weights_normalized + 1  # +1 to ensure all weights are positive

# Features and target
features_train = train_data[['age', 'H', '2B', '3B', 'HR']]
target_train = train_data['total']
features_test = test_data[['age', 'H', '2B', '3B', 'HR']]
target_test = test_data['total']

# Train the model with combined weights
model = LinearRegression()
model.fit(features_train, target_train, sample_weight=combined_weights)

# Predictions
predictions = model.predict(features_test)

# Evaluation
rmse = sqrt(mean_squared_error(target_test, predictions))
r_squared = r2_score(target_test, predictions)

# Predicted data into DataFrame
predictions_df = test_data.copy()
predictions_df['Predicted Hits'] = predictions
predictions_df = predictions_df[['Name', 'Year', 'age', 'H', '2B', '3B', 'HR', 'total', 'Predicted Hits']]

print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")

# Save the predictions DataFrame
predictions_df.to_csv("predictionsWeightedByYearAndAge.csv", index=False)