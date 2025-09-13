import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
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
OUTPUT_GRAPH_PATH = "optimized_rating_graph.png"

# --- Names ---
PLAYER_NAME = "player" 
ORACLE_ENGINE_NAME = "stockfish_full_1" 

# --- Engines to Exclude from Optimization & Final Graph ---
ENGINES_TO_EXCLUDE = []

# --- Original Weights Mapping ---
SCORE_TO_RANK_MAP = {
    1.0: 0,
    0.5: 1,
    0.25: 2,
    0.0: -1
}

# --- Optimization Settings ---
WEIGHT_STEP = 0.05

# ==============================================================================
# --- Core Logic ---
# ==============================================================================

def load_and_prepare_data():
    """Loads log and engine data, and maps scores to ranks."""
    print("--- Loading and Preparing Data ---")
    try:
        log_df = pd.read_csv(GRANULAR_LOG_PATH)
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"'{GRANULAR_LOG_PATH}' is empty. No data to analyze.", file=sys.stderr)
        sys.exit(1)

    # Note: Filtering is now handled inside each specific function for clarity.
    print(f"Loaded {len(log_df)} total log entries.")

    print("Mapping original scores to move ranks...")
    log_df['move_rank'] = log_df['score'].map(SCORE_TO_RANK_MAP).fillna(-1).astype(int)
    print("Mapping complete.")
    
    return log_df, engines_df

def run_optimization(log_df, engines_df):
    """Finds the optimal weights for move ranks based on the log data."""
    print("\n--- Running Weight Optimization ---")
    
    # Create a clean list of benchmark engines, excluding player, oracle, and specified exclusions.
    benchmark_names = engines_df[
        ~engines_df['engine_name'].isin([PLAYER_NAME, ORACLE_ENGINE_NAME] + ENGINES_TO_EXCLUDE)
    ]['engine_name'].unique()
    
    engine_log_df = log_df[log_df['engine_name'].isin(benchmark_names)].copy()
    benchmark_info_df = engines_df[engines_df['engine_name'].isin(benchmark_names)].copy()
      
    weight_range = np.arange(0, 1 + WEIGHT_STEP, WEIGHT_STEP)
    all_combinations = list(product(weight_range, repeat=3))
    weight_combinations = [c for c in all_combinations if c[0] >= c[1] >= c[2]]
    print(f"Generated {len(weight_combinations)} valid weight combinations to test.")

    best_r2 = -1
    best_weights = (0, 0, 0)

    with tqdm(weight_combinations, desc="Optimizing Weights", unit="combo", ncols=100) as pbar:
        for weights in pbar:
            def get_temp_score(rank):
                return 0.0 if rank == -1 else weights[rank]
            
            engine_log_df['temp_score'] = engine_log_df['move_rank'].apply(get_temp_score)
            avg_scores = engine_log_df.groupby('engine_name')['temp_score'].mean().reset_index()
            merged = pd.merge(benchmark_info_df, avg_scores, on='engine_name')
            
            if len(merged) < 2 or merged['temp_score'].nunique() < 2:
                continue

            _s, _i, r_val, _p, _e = linregress(merged['rating'], merged['temp_score'])
            r_squared = r_val**2
            
            if r_squared > best_r2:
                best_r2 = r_squared
                best_weights = tuple(round(w, 2) for w in weights)
                pbar.set_description(f"Optimizing Weights (Best R²: {best_r2:.4f})")
    
    print("\n--- Optimization Complete ---")
    print(f"Best Weights Found: {best_weights}")
    print(f"Highest R-squared during optimization: {best_r2:.6f}")
    return list(best_weights)

def analyze_and_graph_with_new_weights(log_df, engines_df, optimal_weights):
    """Recalculates scores with optimal weights and generates the final graph."""
    print("\n--- Applying Optimal Weights and Generating Graph ---")

    def get_optimal_score(rank):
        return 0.0 if rank == -1 else optimal_weights[rank]
    log_df['optimized_score'] = log_df['move_rank'].apply(get_optimal_score)
    
    avg_scores_df = log_df.groupby('engine_name')['optimized_score'].mean().reset_index()
    avg_scores_df.rename(columns={'optimized_score': 'average_hit_score'}, inplace=True)
    
    player_avg_score_row = avg_scores_df[avg_scores_df['engine_name'] == PLAYER_NAME]
    
    if player_avg_score_row.empty:
        print("No player data found to plot.")
        return
    player_avg_score = player_avg_score_row.iloc[0]['average_hit_score']

    # FIX: Create the final dataframe using only the benchmark engines to ensure
    # the regression is not skewed by the Oracle or player data.
    benchmark_names = engines_df[
        ~engines_df['engine_name'].isin([PLAYER_NAME, ORACLE_ENGINE_NAME] + ENGINES_TO_EXCLUDE)
    ]['engine_name'].unique()
    
    benchmark_info_df = engines_df[engines_df['engine_name'].isin(benchmark_names)]
    engine_avg_scores = avg_scores_df[avg_scores_df['engine_name'].isin(benchmark_names)]
    
    final_df = pd.merge(benchmark_info_df, engine_avg_scores, on='engine_name')

    if len(final_df) < 2 or final_df['average_hit_score'].nunique() < 2:
        print("Not enough benchmark data to create a reliable estimate or graph.")
        return

    slope, intercept, r_value, _, _ = linregress(final_df['rating'], final_df['average_hit_score'])
    r_squared = r_value**2
    
    if slope != 0:
        estimated_player_rating = (player_avg_score - intercept) / slope
    else:
        estimated_player_rating = 0 

    print(f"\n--- Final Analysis with Optimal Weights ---")
    print(f"Player's New Average Hit Score: {player_avg_score:.4f}")
    print(f"Final Correlation (R-squared): {r_squared:.4f}")
    print(f"Final Estimated Player Rating: {estimated_player_rating:.0f}")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
    sns.regplot(x='rating', y='average_hit_score', data=final_df, ci=95,
                line_kws={'color': 'red', 'linestyle': '--'},
                scatter_kws={'s': 80, 'alpha': 0.8})

    plt.title(f'Hit Score vs. Rating (Optimized Weights: {tuple(optimal_weights)})', fontsize=16)
    plt.xlabel('Engine Rating (Elo)', fontsize=12)
    plt.ylabel('Average Hit Score (Recalculated)', fontsize=12)
    
    plt.scatter(estimated_player_rating, player_avg_score, color='gold', 
                s=200, edgecolor='black', zorder=5, label=f'You ({PLAYER_NAME_IN_PGN})')
    
    for _, row in final_df.iterrows():
        plt.text(row['rating'] + 10, row['average_hit_score'], row['engine_name'], fontsize=9)
    
    plt.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$', transform=plt.gca().transAxes, 
             fontsize=14, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    
    plt.legend(loc='lower right')
    
    plt.savefig(OUTPUT_GRAPH_PATH)
    print(f"\nSuccessfully generated optimized graph: '{OUTPUT_GRAPH_PATH}'")

def main():
    """Main function to run the full pipeline."""
    # A global variable for the player's PGN name for the plot label
    global PLAYER_NAME_IN_PGN
    PLAYER_NAME_IN_PGN = "Desjardins373"

    log_df, engines_df = load_and_prepare_data()
    optimal_weights = run_optimization(log_df, engines_df)
    analyze_and_graph_with_new_weights(log_df, engines_df, optimal_weights)

if __name__ == "__main__":
    main()
