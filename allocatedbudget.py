import pandas as pd

def allocate_budget_weighted_top_player(free_agents, budget, max_players):
    """
    Allocate budget to 7 unique players, ensuring the top player receives a significantly 
    higher portion of the budget. The remaining budget is distributed among the next six players.
    """
    # Ensure 'Allocated Budget' column is of type float
    free_agents['Allocated Budget'] = 0.0
    
    # Select the top player by WAR for each position
    top_players_by_position = free_agents.loc[free_agents.groupby('pos')['Estimated WAR'].idxmax()]
    
    # Ensure we only consider up to the specified max number of players, prioritized by WAR
    top_players_by_position = top_players_by_position.sort_values(by='Estimated WAR', ascending=False).head(max_players)
    
    # Allocate a significant portion of the budget to the top player
    top_player_allocation = budget * 0.35  # For example, 35% of the budget goes to the top player
    top_players_by_position.iloc[0, top_players_by_position.columns.get_loc('Allocated Budget')] = top_player_allocation
    
    # Distribute the remaining budget among the next six players based on their WAR
    remaining_budget = budget - top_player_allocation
    if len(top_players_by_position) > 1:
        total_war_remaining_players = top_players_by_position.iloc[1:]['Estimated WAR'].sum()
        for index, row in top_players_by_position.iloc[1:].iterrows():
            player_share_of_war = row['Estimated WAR'] / total_war_remaining_players
            allocation = remaining_budget * player_share_of_war
            top_players_by_position.at[index, 'Allocated Budget'] = allocation
    
    return top_players_by_position

# Assuming df_free_agents is loaded with the correct path
df_free_agents = pd.read_csv('sorted_by_estimated_war.csv')

# Apply the budget allocation function
allocated_budget_df = allocate_budget_weighted_top_player(df_free_agents, 200, 7)

# Display the final allocation
print(allocated_budget_df[['Name', 'pos', 'Estimated WAR', 'Allocated Budget']])

# Save the allocated budget DataFrame to a new CSV file
allocated_budget_df.to_csv('weighted_top_player_allocated_budget_free_agents.csv', index=False)
