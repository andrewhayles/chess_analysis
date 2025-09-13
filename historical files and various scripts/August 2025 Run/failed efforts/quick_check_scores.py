import pandas as pd
import os
import sys
from scipy.stats import linregress

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- File Paths (Should match your main script) ---
GRANULAR_LOG_PATH = "granular_analysis_log.csv"
ENGINES_CSV_PATH = "real_engines.csv"
PLAYER_NAME_IN_PGN = "Desjardins373" # Used for display purposes

# ==============================================================================
# --- Main Calculation Logic ---
# ==============================================================================

def calculate_and_report():
    """
    Loads the granular log, calculates the current average scores for all
    entities, and provides a rating estimate based on the available data.
    """
    print("--- Calculating Current Progress from Granular Log ---")

    # 1. Load the granular log file
    if not os.path.exists(GRANULAR_LOG_PATH):
        print(f"Error: Granular log file not found at '{GRANULAR_LOG_PATH}'.")
        print("Please run the main analysis script first to generate some data.")
        return

    try:
        log_df = pd.read_csv(GRANULAR_LOG_PATH)
        if log_df.empty:
            print("Log file is empty. No data to analyze yet.")
            return
    except pd.errors.EmptyDataError:
        print("Log file is empty. No data to analyze yet.")
        return

    # 2. Load engine ratings
    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Engines CSV file not found at '{ENGINES_CSV_PATH}'. Cannot calculate rating.")
        return

    # 3. Calculate average scores for all entities in the log
    print(f"Found {len(log_df)} total analyses in the log file.\n")
    avg_scores = log_df.groupby('engine_name')['score'].mean().reset_index()
    avg_scores.rename(columns={'score': 'average_score'}, inplace=True)

    print("--- Current Average Scores ---")
    # Sort by engine name for consistent display
    for _, row in avg_scores.sort_values(by='engine_name').iterrows():
        print(f"  - {row['engine_name']:<20}: {row['average_score']:.4f}")
    
    # 4. Separate player and engine scores for regression
    player_score_row = avg_scores[avg_scores['engine_name'] == 'player']
    engine_scores = avg_scores[avg_scores['engine_name'] != 'player']

    if player_score_row.empty:
        print("\nNo data for the player found yet. Cannot provide a rating estimate.")
        return
        
    player_avg_score = player_score_row.iloc[0]['average_score']

    # Merge with engine ratings
    merged_engines = pd.merge(engine_scores, engines_df, on='engine_name')

    # 5. Perform linear regression and estimate rating
    print("\n--- Current Rating Estimate ---")
    
    # Check if there's enough data to create a trend line
    if len(merged_engines) < 2:
        print("Not enough benchmark engine data (need at least 2) to create a rating estimate.")
        return
    
    # Check for variance in scores to avoid division by zero
    if merged_engines['average_score'].nunique() < 2:
        print("Cannot estimate rating: all benchmark engines have the same average score.")
        return

    slope, intercept, r_value, _, _ = linregress(merged_engines['average_score'], merged_engines['rating'])
    r_squared = r_value**2
    estimated_rating = (slope * player_avg_score) + intercept

    print(f"Player's Current Average Score: {player_avg_score:.4f}")
    print(f"Based on {len(merged_engines)} benchmark engines.")
    print(f"Current R-squared: {r_squared:.4f}")
    print(f"Current Estimated Rating for {PLAYER_NAME_IN_PGN}: {estimated_rating:.0f}")
    print("\n--- Script Finished ---")


if __name__ == "__main__":
    calculate_and_report()
