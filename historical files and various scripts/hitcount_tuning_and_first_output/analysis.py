import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import os

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- File Paths ---
# Make sure these paths point to your log file and your engines list.
DETAILED_LOG_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/hitcount_tuning/real_time_analysis_log.csv"
ENGINES_CSV_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/real_engines.csv"

# --- Analysis Configuration ---
# List the specific 'run_name' values from the log you want to analyze.
RUNS_TO_ANALYZE = [
    "Top 1 Move Only",
    "Top 3 Moves, Linear Weights"
]

# Directory to save the output graphs
OUTPUT_DIR = "C:/Users/desja/Documents/Python_programs/chess_study/hitcount_tuning/analysis_graphs"

# ==============================================================================
# --- Analysis and Visualization Logic ---
# ==============================================================================

def analyze_log_file():
    """
    Reads the real-time log file, calculates R-squared for specified runs,
    and generates regression plots.
    """
    print("--- Starting Log Analysis Script ---")

    # --- Load Data ---
    try:
        print(f"Loading log file from: {DETAILED_LOG_PATH}")
        log_df = pd.read_csv(DETAILED_LOG_PATH)
    except FileNotFoundError:
        print(f"Error: Log file not found at '{DETAILED_LOG_PATH}'. Please check the path.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Log file at '{DETAILED_LOG_PATH}' is empty. No data to analyze.")
        return

    try:
        print(f"Loading engine data from: {ENGINES_CSV_PATH}")
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
        if 'rating' not in engines_df.columns or 'engine_name' not in engines_df.columns:
            raise ValueError("Engines CSV must contain 'rating' and 'engine_name' columns.")
    except FileNotFoundError:
        print(f"Error: Engines CSV file not found at '{ENGINES_CSV_PATH}'.")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output graphs will be saved to: {OUTPUT_DIR}")

    # --- Process Each Specified Run ---
    for run_name in RUNS_TO_ANALYZE:
        print(f"\n--- Analyzing Run: '{run_name}' ---")
        
        # Filter the log for the specific run
        run_df = log_df[log_df['run_name'] == run_name]

        if run_df.empty:
            print(f"No data found for run '{run_name}' in the log file. Skipping.")
            continue

        # Calculate the average score for each engine in this run
        avg_scores = run_df.groupby('player_engine')['score'].mean().reset_index()
        avg_scores.rename(columns={'player_engine': 'engine_name', 'score': 'average_hit_score'}, inplace=True)
        
        # Merge with engine ratings
        final_df = pd.merge(engines_df, avg_scores, on='engine_name')
        
        if len(final_df) < 2:
            print("Not enough data points (< 2) to create a correlation. Skipping graph.")
            continue

        # --- Calculate R-squared ---
        lin_reg_result = linregress(final_df['rating'], final_df['average_hit_score'])
        r_squared = lin_reg_result.rvalue ** 2
        
        print(f"Calculated R-squared value: {r_squared:.4f}")

        # --- Generate and Save Graph ---
        plt.figure(figsize=(12, 7))
        sns.regplot(x='rating', y='average_hit_score', data=final_df, line_kws={"color": "red"})
        
        plt.title(f"Hit Score vs. Engine Rating\n(Method: {run_name})", fontsize=16)
        plt.xlabel("Engine Rating", fontsize=12)
        plt.ylabel("Average Hit Score", fontsize=12)
        plt.grid(True)
        
        # Add R-squared value to the plot
        plt.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$', transform=plt.gca().transAxes,
                 fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
                 
        # Save the figure
        graph_file_name = f"{run_name.replace(' ', '_').lower()}_correlation.png"
        graph_path = os.path.join(OUTPUT_DIR, graph_file_name)
        plt.savefig(graph_path)
        plt.close()
        
        print(f"Graph saved to: {graph_path}")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    analyze_log_file()
