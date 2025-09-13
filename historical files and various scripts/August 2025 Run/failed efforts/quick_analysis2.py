import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ==============================================================================
# --- Configuration ---
# ==============================================================================
GRANULAR_LOG_PATH = "granular_analysis_log.csv"
ENGINES_CSV_PATH = "real_engines.csv"
OUTPUT_GRAPH_PATH = "live_progress_graph.png"
PLAYER_NAME = "Desjardins373"

ENGINES_TO_EXCLUDE = []

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================
def logistic(x, L, k, x0):
    """Logistic function for curve fitting."""
    return L / (1 + np.exp(-k * (x - x0)))

# ==============================================================================
# --- Core Logic ---
# ==============================================================================
def analyze_and_graph():
    print("--- Live Analysis Visualizer (Logistic) ---")

    if not os.path.exists(GRANULAR_LOG_PATH):
        print(f"Error: Granular log file not found at '{GRANULAR_LOG_PATH}'.")
        return
    if not os.path.exists(ENGINES_CSV_PATH):
        print(f"Error: Engines CSV file not found at '{ENGINES_CSV_PATH}'.")
        return

    try:
        log_df = pd.read_csv(GRANULAR_LOG_PATH)
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except pd.errors.EmptyDataError:
        print(f"'{GRANULAR_LOG_PATH}' is empty. No data to analyze yet.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV files: {e}")
        return

    if log_df.empty:
        print("Log file is empty. No data to analyze yet.")
        return

    print(f"Loaded {len(log_df)} total analyses from the log file.")

    # Exclude unwanted engines
    if ENGINES_TO_EXCLUDE:
        print(f"Excluding the following engines: {', '.join(ENGINES_TO_EXCLUDE)}")
        log_df = log_df[~log_df['engine_name'].isin(ENGINES_TO_EXCLUDE)]
        engines_df = engines_df[~engines_df['engine_name'].isin(ENGINES_TO_EXCLUDE)]
        if log_df.empty:
            print("No data left after excluding engines. Cannot proceed.")
            return

    # Calculate average scores
    avg_scores_df = log_df.groupby('engine_name')['score'].mean().reset_index()
    avg_scores_df.rename(columns={'score': 'average_hit_score'}, inplace=True)

    player_avg_score_row = avg_scores_df[avg_scores_df['engine_name'] == 'player']
    engine_avg_scores = avg_scores_df[avg_scores_df['engine_name'] != 'player']

    if player_avg_score_row.empty:
        print("No analysis for the player found in the log yet.")
        return
    player_avg_score = player_avg_score_row.iloc[0]['average_hit_score']

    benchmark_engine_info = engines_df[engines_df['engine_name'] != 'stockfish_full_1']
    final_df = pd.merge(benchmark_engine_info, engine_avg_scores, on='engine_name')

    if len(final_df) < 2:
        print("Not enough benchmark engine data in the log to create a reliable estimate.")
        return

    if final_df['average_hit_score'].nunique() < 2:
        print("Cannot generate a graph: all benchmark engines have the same average score.")
        return

    # --- Logistic Regression ---
    try:
        # Fit logistic curve
        xdata = final_df['rating'].values
        ydata = final_df['average_hit_score'].values
        p0 = [1.0, 0.005, 1800]  # initial guesses: L, k, x0
        popt, _ = curve_fit(logistic, xdata, ydata, p0, maxfev=10000)
        L, k, x0 = popt

        # Goodness of fit (R²)
        predicted = logistic(xdata, *popt)
        residuals = ydata - predicted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata - np.mean(ydata))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Estimate player's rating from their hit score
        est_player_rating = x0 + (1/k) * np.log((L/player_avg_score) - 1)

        print(f"\n--- Current Overall Status (Logistic) ---")
        print(f"Based on data from {log_df['fen'].nunique()} positions.")
        print(f"Player's Average Hit Score: {player_avg_score:.4f}")
        print(f"Correlation (R-squared): {r_squared:.4f}")
        print(f"Estimated Player Rating: {est_player_rating:.0f}")

        # --- Plot ---
        plt.figure(figsize=(12, 8))

        # Logistic curve
        x_vals = np.linspace(min(xdata), max(xdata), 500)
        y_vals = logistic(x_vals, *popt)
        plt.plot(x_vals, y_vals, 'r--', label='Logistic Fit')

        # Scatter engine points
        plt.scatter(final_df['rating'], final_df['average_hit_score'], s=80, alpha=0.8)
        for _, row in final_df.iterrows():
            plt.text(row['rating'] + 10, row['average_hit_score'], row['engine_name'], fontsize=9)

        # Player point
        plt.scatter(est_player_rating, player_avg_score, color='gold', s=200, edgecolor='black', zorder=5, label=f'You ({PLAYER_NAME})')
        plt.text(est_player_rating + 10, player_avg_score, f'You ({est_player_rating:.0f})', fontsize=11, weight='bold')

        plt.title('Hit Score vs. Engine Rating (Live Progress, Logistic)', fontsize=16)
        plt.xlabel('Engine Rating (Elo)', fontsize=12)
        plt.ylabel('Average Hit Score', fontsize=12)
        plt.legend()
        plt.grid(True)

        plt.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$', transform=plt.gca().transAxes, fontsize=14,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))

        plt.savefig(OUTPUT_GRAPH_PATH)
        print(f"\nSuccessfully generated live progress graph: '{OUTPUT_GRAPH_PATH}'")

    except RuntimeError:
        print("Logistic regression failed to converge. Consider checking input data or fall back to linear.")

if __name__ == "__main__":
    analyze_and_graph()
