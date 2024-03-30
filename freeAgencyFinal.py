import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import numpy as np


# Load the dataset
df = pd.read_csv('sorted_filtered_batting_season_summary.csv')
df_sorted = df.sort_values(by='Year')

# Define cutoff year and split the dataset
cutoff_year = df_sorted['Year'].quantile(0.8, interpolation='higher')
train_data = df_sorted[df_sorted['Year'] <= cutoff_year]
test_data = df_sorted[df_sorted['Year'] > cutoff_year]

# Specify the features and target
features_cols = ['age', 'AB', 'BB', 'SO', 'SLG', 'OPS', 'BA']  # Adjust based on your dataset
features_train = train_data[features_cols]
target_train = train_data['total']

# Prepare year weights - more recent years should have higher weights
max_year = df_sorted['Year'].max()
min_year = df_sorted['Year'].min()
df_sorted['year_weight'] = (df_sorted['Year'] - min_year) / (max_year - min_year)

# Identify each player's prime year and calculate prime-based weights
prime_years = df_sorted.loc[df_sorted.groupby('Name')['total'].idxmax()]
prime_years['prime_weight'] = 1  # Max weight for prime years
df_sorted = pd.merge(df_sorted, prime_years[['Name', 'Year', 'prime_weight']], on=['Name', 'Year'], how='left')
df_sorted['prime_weight'].fillna(0.5, inplace=True)  # Lower weight for non-prime years

# Combine year_weight and prime_weight for final sample weights
df_sorted['sample_weight'] = df_sorted['year_weight'] + df_sorted['prime_weight']
train_weights = df_sorted.loc[df_sorted.index.isin(train_data.index), 'sample_weight']

# Create a pipeline with data preprocessing and modeling steps
scaler = StandardScaler()
pca = PCA(n_components=0.95)  # Keep 95% of variance
model = LinearRegression()
pipe = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('linear_regression', model)])

# Fit the model using the training set and sample weights
pipe.fit(features_train, target_train, linear_regression__sample_weight=train_weights)

# Predict on test data and evaluate
features_test = test_data[features_cols]
target_test = test_data['total']
predictions = pipe.predict(features_test)
rmse = sqrt(mean_squared_error(target_test, predictions))
r_squared = r2_score(target_test, predictions)

print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")

# Prepare next year's data for prediction
next_year_data = test_data.copy()
next_year_data['age'] += 1

# Predict the expected number of hits for next year
features_next_year = next_year_data[features_cols]
predicted_hits_next_year = pipe.predict(features_next_year)

# Create a DataFrame with player names and expected hits for next year
predictions_for_next_year_df = next_year_data[['Name']].copy()
predictions_for_next_year_df['Expected Number of Hits Next Season'] = predicted_hits_next_year.round().astype(int)

# Assuming df_free_agents is loaded with the correct path
df_free_agents = pd.read_csv('sorted_by_estimated_war.csv')

# Apply the budget allocation function

print(predictions_for_next_year_df.head())

predictions_for_next_year_df.to_csv('expected_hits_for_next_year.csv', index=False)

# Data Visualization for Linear Regression Model
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the following variables are defined in your script: df_sorted, features_train, target_train, predictions, target_test

# Distribution of the Target Variable
plt.figure(figsize=(10, 6))
sns.histplot(df_sorted['total'], kde=True)
plt.title('Distribution of Total Hits')
plt.xlabel('Total Hits')
plt.ylabel('Frequency')
plt.show()

# Correlation Matrix Heatmap
corr_matrix = df_sorted[features_cols + ['total']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Features Correlation Matrix')
plt.show()

# PCA Variance Explained
# Note: This section assumes PCA has been fitted as part of your pipeline
pca_variance = pipe.named_steps['pca'].explained_variance_ratio_
plt.figure(figsize=(8, 5))
sns.barplot(x=list(range(1, len(pca_variance) + 1)), y=pca_variance)
plt.title('PCA Variance Explained')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.show()

# Actual vs. Predicted Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(target_test, predictions, alpha=0.5)
plt.plot([target_test.min(), target_test.max()], [target_test.min(), target_test.max()], 'k--', lw=4)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values for Total Hits')
plt.show()

# Residual Plot
residuals = target_test - predictions
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals)
plt.hlines(y=0, xmin=predictions.min(), xmax=predictions.max(), colors='red', linestyles='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals of Predictions')
plt.show()
