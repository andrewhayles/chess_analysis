import pandas as pd
import numpy as np
from scipy.stats import linregress
from itertools import product
from tqdm import tqdm
import os
import sys

# ==============================================================================
# --- Configuration ---
# ==============================================================================
# --- File Paths ---
GRANULAR_LOG_PATH = "granular_analysis_log.csv"
ENGINES_CSV_PATH = "real_engines.csv"

# --- Player Name ---
# This is used to separate player data from engine data.
PLAYER_NAME = "Desjardins373"

# --- Engines to Exclude ---
# Add the names of any engines you want to exclude from the optimization.
# For example: ENGINES_TO_EXCLUDE = ["dragon", "stockfish_full_test"]
ENGINES_TO_EXCLUDE = []

# --- Original Weights Mapping ---
# This maps the score from your log file back to the rank of the move that was played.
# This MUST match the weights used in your rating_estimate12.py script.
SCORE_TO_RANK_MAP = {
    1.0: 0,  # A score of 1.0 means it was the 1st best move (index 0)
    0.5: 1,  # A score of 0.5 means it was the 2nd best move (index 1)
    0.25: 2, # A score of 0.25 means it was the 3rd best move (index 2)
    0.0: -1  # A score of 0.0 means it was another move (we'll use -1 as a placeholder)
}

# --- Optimization Settings ---
# The script will test weights from 0.0 to 1.0 in steps of this value.
WEIGHT_STEP = 0.05

# ==============================================================================
# --- Core Logic ---
# ==============================================================================

def load_data():
    """Loads and filters the log and engine data files."""
    try:
        log_df = pd.read_csv(GRANULAR_LOG_PATH)
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e}. Exiting.", file=sys.stderr)
        return None, None
    except pd.errors.EmptyDataError:
        print(f"'{GRANULAR_LOG_PATH}' is empty. No data to analyze.", file=sys.stderr)
        return None, None
    
    # --- Filter out excluded engines ---
    if ENGINES_TO_EXCLUDE:
        print(f"Excluding the following engines from optimization: {', '.join(ENGINES_TO_EXCLUDE)}")
        log_df = log_df[~log_df['engine_name'].isin(ENGINES_TO_EXCLUDE)]
        engines_df = engines_df[~engines_df['engine_name'].isin(ENGINES_TO_EXCLUDE)]
        if log_df.empty or engines_df.empty:
            print("No data left after excluding engines. Cannot proceed.")
            return None, None
            
    # Exclude the oracle engine and the player from the benchmark set for regression
    benchmark_engines_df = engines_df[
        ~engines_df['engine_name'].str.contains('stockfish_full', case=False) &
        (engines_df['engine_name'] != PLAYER_NAME)
    ].copy()
    
    return log_df, benchmark_engines_df

def calculate_r_squared(log_df_with_ranks, new_weights, benchmark_info_df):
    """Calculates R-squared for a given set of new weights using the pre-calculated ranks."""
    
    def get_new_score(rank):
        if rank == -1: # Move was not in the top 3
            return 0.0
        return new_weights[rank]

    # Create a new score column based on the move rank and the new weights
    log_df_with_ranks['recalculated_score'] = log_df_with_ranks['move_rank'].apply(get_new_score)
    
    # Calculate the average of the *new* scores for each engine
    avg_scores_df = log_df_with_ranks.groupby('engine_name')['recalculated_score'].mean().reset_index()
    
    # Merge with engine ratings
    merged_df = pd.merge(benchmark_info_df, avg_scores_df, on='engine_name')
    
    # Need at least 2 data points for a regression
    if len(merged_df) < 2 or merged_df['recalculated_score'].nunique() < 2:
        return -1 # Not enough data or no variance in scores

    # Perform linear regression: rating (x) vs. recalculated_score (y)
    _slope, _intercept, r_value, _p_value, _std_err = linregress(merged_df['rating'], merged_df['recalculated_score'])
    
    return r_value**2

def main():
    """Main function to run the fast, log-based weight optimization."""
    print("--- Fast Log-Based Weight Optimizer ---")
    
    log_df, benchmark_info_df = load_data()
    if log_df is None:
        return

    print(f"Loaded {len(log_df)} relevant log entries.")

    # --- Pre-computation Step ---
    # Map the original scores to move ranks. This is done only once.
    print("Mapping original scores to move ranks...")
    log_df['move_rank'] = log_df['score'].map(SCORE_TO_RANK_MAP)
    
    # Handle any scores that might not be in the map
    log_df['move_rank'].fillna(-1, inplace=True) 
    log_df['move_rank'] = log_df['move_rank'].astype(int)
    
    # Separate engine data for the calculations
    engine_log_df = log_df[log_df['engine_name'] != 'player'].copy()
    print("Mapping complete.")

    # --- Optimization Loop ---
    # Generate all possible weight combinations
    weight_range = np.arange(0, 1 + WEIGHT_STEP, WEIGHT_STEP)
    all_combinations = list(product(weight_range, repeat=3))
    # Filter for combinations where weights are in descending order
    weight_combinations = [combo for combo in all_combinations if combo[0] >= combo[1] >= combo[2]]
    print(f"Generated {len(weight_combinations)} valid weight combinations to test.")

    best_r2 = -1
    best_weights = (0, 0, 0)

    # Iterate through all combinations with a progress bar
    with tqdm(weight_combinations, desc="Optimizing Weights", unit="combo", ncols=100) as pbar:
        for weights in pbar:
            # The weights need to be a list for indexing
            r_squared = calculate_r_squared(engine_log_df, list(weights), benchmark_info_df)
            
            if r_squared > best_r2:
                best_r2 = r_squared
                best_weights = tuple(round(w, 2) for w in weights)
                pbar.set_description(f"Optimizing Weights (Best R²: {best_r2:.4f})")

    # --- Display Final Results ---
    print("\n" + "="*60)
    print("Optimization Complete!")
    print("\n--- Optimal Result ---")
    print(f"Best Weights (1st, 2nd, 3rd): {best_weights}")
    print(f"Highest R-squared: {best_r2:.6f}")
    print("="*60)

if __name__ == "__main__":
    main()
