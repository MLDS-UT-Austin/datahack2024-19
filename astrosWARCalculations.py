import pandas as pd

# Load the dataset
df = pd.read_csv('astros_2023_players.csv')

#Calculating Offensive Score
df['Offensive Score'] = (df['OBP'] + df['SLG']) * df['PA']

# Estimate WAR
league_average_offensive_score = df['Offensive Score'].mean()
df['Estimated WAR'] = (df['Offensive Score'] - league_average_offensive_score) / (league_average_offensive_score / 10)

# Sort players by WAR
df_sorted_by_estimated_war = df.sort_values(by='Estimated WAR', ascending=False)

# Select relevant columns to display including WAR
columns_to_display = ['Name', 'Year', 'OBP', 'SLG', 'PA', 'Estimated WAR']
print(df_sorted_by_estimated_war[columns_to_display].head())

# Optionally, save this sorted list to a new CSV
df_sorted_by_estimated_war.to_csv('sorted_astros_by_estimated_war.csv', index=False)