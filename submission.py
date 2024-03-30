import pandas as pd

# Load the dataset with allocated budgets
allocated_budget_df = pd.read_csv('weighted_top_player_allocated_budget_free_agents.csv')

# Load the dataset with expected hits for next year
expected_hits_df = pd.read_csv('expected_hits_for_next_year.csv')

# Merge the allocated budget into the expected hits data
# This adds the 'Allocated Budget' to the expected hits dataframe based on matching player names
expected_hits_with_budget = pd.merge(
    expected_hits_df, 
    allocated_budget_df[['Name', 'Allocated Budget']], 
    on='Name', 
    how='left'
)

# For players not found in the allocated budget list, we set their 'Allocated Budget' to 0.0
expected_hits_with_budget['Allocated Budget'].fillna(0.0, inplace=True)

# Rename 'Allocated Budget' to 'Bid Amount($)' for clarity
expected_hits_with_budget.rename(columns={'Allocated Budget': 'Bid Amount($)'}, inplace=True)

# Sort the DataFrame by 'Bid Amount($)' from greatest to least
expected_hits_with_budget_sorted = expected_hits_with_budget.sort_values(by='Bid Amount($)', ascending=False)

# Save the sorted DataFrame to a new CSV file
expected_hits_with_budget_sorted.to_csv('submission.csv', index=False)

print("The expected hits data with allocated budget has been sorted by bid amount and saved.")
