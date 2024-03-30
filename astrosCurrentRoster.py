import pandas as pd

# Assuming the dataset is named 'batting_logs.csv' and includes 'Team' and 'Year' columns
df = pd.read_csv('example_data/batting_season_summary.csv')

# Filter the DataFrame for Houston Astros players in 2023
astros_2023 = df[(df['team'] == 'Astros') & (df['Year'] == 2023)]

astros_2023.to_csv('astros_2023_players.csv', index=False)


# You can now proceed to work with 'astros_2023' DataFrame, which is filtered as needed
print(astros_2023.head())