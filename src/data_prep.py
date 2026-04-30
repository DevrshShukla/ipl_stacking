import pandas as pd
import numpy as np
import os

def prepare_data(matches_path='../matches.csv', deliveries_path='../deliveries.csv', output_dir='../data'):
    print("Loading data...")
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    
    print("Merging data...")
    # deliveries has match_id, matches has id
    df = deliveries.merge(matches, left_on='match_id', right_on='id', how='inner')
    
    # We only care about the first innings
    df = df[df['inning'] == 1]
    
    print("Filtering teams...")
    active_teams = [
        'Chennai Super Kings',
        'Delhi Capitals',
        'Gujarat Titans',
        'Kolkata Knight Riders',
        'Lucknow Super Giants',
        'Mumbai Indians',
        'Punjab Kings',
        'Rajasthan Royals',
        'Royal Challengers Bangalore',
        'Sunrisers Hyderabad'
    ]
    
    # Mapping old names to new names where appropriate
    team_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad', # rough map
        'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'
    }
    
    df['batting_team'] = df['batting_team'].replace(team_mapping)
    df['bowling_team'] = df['bowling_team'].replace(team_mapping)
    
    # Filter only active teams
    df = df[(df['batting_team'].isin(active_teams)) & (df['bowling_team'].isin(active_teams))]
    
    # Filter out rain affected matches (where method is not NA, though D/L is usually 1 or NA in some datasets)
    # Let's check if 'method' column has D/L
    if 'method' in df.columns:
        df = df[df['method'].isna() | (df['method'] != 'D/L')]
        
    print("Feature Engineering...")
    # Calculate current score
    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    
    # Calculate wickets fallen
    # is_wicket might be 1 or 0, or we check player_dismissed
    df['is_wicket'] = df['player_dismissed'].apply(lambda x: 1 if pd.notna(x) and x != 'NA' and str(x).strip() != '' else 0)
    df['wickets'] = df.groupby('match_id')['is_wicket'].cumsum()
    
    # Calculate overs completed
    # 'over' is 0-indexed, 'ball' is 1-6
    df['overs_completed'] = df['over'] + df['ball'] / 6.0
    
    # We want to predict the final total score for each match
    final_scores = df.groupby('match_id')['total_runs'].sum().reset_index()
    final_scores.rename(columns={'total_runs': 'final_score'}, inplace=True)
    
    df = df.merge(final_scores, on='match_id', how='inner')
    
    # Select relevant columns for the model
    # We want context after at least 5 overs
    df = df[df['overs_completed'] >= 5.0]
    
    relevant_columns = [
        'match_id', 'batting_team', 'bowling_team', 'venue', 
        'current_score', 'overs_completed', 'wickets', 'final_score'
    ]
    
    # Check if 'venue' is present
    if 'venue' not in df.columns and 'city' in df.columns:
        df['venue'] = df['city']
        
    venue_mapping = {
        'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
        'Feroz Shah Kotla': 'Arun Jaitley Stadium',
        'Brabourne Stadium, Mumbai': 'Brabourne Stadium',
        'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
        'Eden Gardens, Kolkata': 'Eden Gardens',
        'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'Himachal Pradesh Cricket Association Stadium',
        'M Chinnaswamy Stadium, Bengaluru': 'M. Chinnaswamy Stadium',
        'M Chinnaswamy Stadium': 'M. Chinnaswamy Stadium',
        'M.Chinnaswamy Stadium': 'M. Chinnaswamy Stadium',
        'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
        'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
        'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium',
        'Subrata Roy Sahara Stadium': 'Maharashtra Cricket Association Stadium',
        'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association Stadium, Mohali',
        'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association Stadium, Mohali',
        'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association Stadium, Mohali',
        'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
        'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium',
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
        'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
        'Sardar Patel Stadium, Motera': 'Narendra Modi Stadium',
        'Narendra Modi Stadium, Ahmedabad': 'Narendra Modi Stadium',
        'Sheikh Zayed Stadium': 'Zayed Cricket Stadium, Abu Dhabi'
    }
    df['venue'] = df['venue'].replace(venue_mapping)

    
    processed_df = df[relevant_columns]
    
    # Drop NaNs
    processed_df = processed_df.dropna()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'processed_data.csv')
    processed_df.to_csv(output_path, index=False)
    print(f"Data prepared and saved to {output_path}")

if __name__ == "__main__":
    prepare_data('matches.csv', 'deliveries.csv', 'data')
