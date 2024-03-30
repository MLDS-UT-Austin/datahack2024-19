import pandas as pd

# Load the datasets
batting_season_summary = pd.read_csv('example_data/batting_season_summary.csv')
submission_example = pd.read_csv('example_data/submission_example.csv')

# Filter the batting_season_summary DataFrame to include only names present in submission_example
filtered_data = batting_season_summary[batting_season_summary['Name'].isin(submission_example['Name'])]

# Sort the filtered data by the 'Name' column in alphabetical order
sorted_filtered_data = filtered_data.sort_values(by=['Name', 'age'])

sorted_filtered_data['total'] = sorted_filtered_data[['H', '2B', '3B', 'HR']].sum(axis=1)


# Save the sorted filtered data to a new CSV file
sorted_filtered_data_path = 'sorted_filtered_batting_season_summary.csv'
sorted_filtered_data.to_csv(sorted_filtered_data_path, index=False)

print(f"Sorted and filtered data saved to {sorted_filtered_data_path}")