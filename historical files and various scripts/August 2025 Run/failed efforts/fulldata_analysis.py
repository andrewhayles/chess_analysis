import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import os

# --- Configuration ---
GRANULAR_LOG_FILE = 'granular_analysis_log_top3.csv'
ENGINES_CSV_FILE = 'real_engines.csv'
PLAYER_ENGINE_NAME = 'Desjardins373' # The engine/player to estimate the rating for
OUTPUT_REPORT_FILE = 'full_rating_analysis_report_with_estimation.txt'
OUTPUT_PLOT_FILE = 'full_rating_analysis_plot_with_estimation.png'

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. Data Loading and Preparation ---

def load_data(log_file, engines_file):
    """Loads the main analysis log and merges it with engine ratings."""
    print(f"Loading granular data from '{log_file}'...")
    if not os.path.exists(log_file):
        print(f"FATAL: Main log file not found: {log_file}")
        return None, None

    print(f"Loading engine ratings from '{engines_file}'...")
    if not os.path.exists(engines_file):
        print(f"FATAL: Engines file not found: {engines_file}")
        return None, None
        
    main_df = pd.read_csv(log_file)
    engines_df = pd.read_csv(engines_file)
    
    engine_ratings = pd.Series(engines_df.rating.values, index=engines_df.engine_name).to_dict()
    
    # Keep Desjardins373 data, but map other engine ratings
    main_df['rating'] = main_df['engine_name'].map(engine_ratings)
    
    print(f"Successfully loaded and merged data: {len(main_df)} records found.")
    return main_df, engine_ratings

# --- 2. Analysis and Modeling ---

def calculate_hit_percentage(df, weights):
    """Calculates the weighted hit percentage for each engine."""
    engine_scores = {}
    for engine, group in df.groupby('engine_name'):
        weighted_hits = group['move_rank'].apply(lambda rank: weights.get(rank, 0)).sum()
        total_possible_score = len(group) * weights.get(0, 1) 
        engine_scores[engine] = (weighted_hits / total_possible_score) * 100 if total_possible_score > 0 else 0
    return engine_scores

def elo_curve(x, a, b, c):
    """Defines the logistic curve used to model Elo ratings."""
    return a / (1 + np.exp(-b * (x - c)))

def fit_elo_curve(ratings, hit_percentages):
    """
    Fits the Elo curve to the data and returns the curve parameters.
    This version includes better initial guesses and bounds for more robust fitting.
    """
    initial_guesses = [max(hit_percentages), 0.01, np.median(ratings)]
    bounds = ([0, 0, min(ratings)], [101, 1, max(ratings)])
    
    try:
        params, _ = curve_fit(elo_curve, ratings, hit_percentages, p0=initial_guesses, bounds=bounds, maxfev=10000)
        return params
    except RuntimeError:
        print("  Warning: Could not find optimal parameters for the Elo curve.")
        return None

# --- NEW: Function to estimate rating from the model ---
def estimate_rating_from_percentage(percentage, params):
    """Estimates an Elo rating by inverting the logistic curve function."""
    a, b, c = params
    # Inverted formula: x = c - (ln(a/y - 1) / b)
    if percentage >= a: # Cannot estimate if percentage is at or above the curve's maximum
        return float('inf')
    if percentage <= 0:
        return float('-inf')
        
    try:
        # Prevents math error if a/percentage is <= 1
        if (a / percentage) - 1 <= 0:
            return float('inf')
        
        estimated_rating = c - (np.log((a / percentage) - 1) / b)
        return estimated_rating
    except (ValueError, ZeroDivisionError):
        return None

# --- 3. Main Execution ---

def main():
    """Main function to run the full analysis and generate reports."""
    start_time = datetime.now()
    print("--- Starting Full Rating Analysis with Estimation ---")
    
    main_df, engine_ratings = load_data(GRANULAR_LOG_FILE, ENGINES_CSV_FILE)
    if main_df is None: return

    weight_scenarios = [
        {'name': 'Top1_Only', 'weights': {0: 1.0, 1: 0.0, 2: 0.0}},
        {'name': 'Balanced', 'weights': {0: 1.0, 1: 0.5, 2: 0.25}},
        {'name': 'Linear_Decay', 'weights': {0: 1.0, 1: 0.66, 2: 0.33}},
    ]
    
    with open(OUTPUT_REPORT_FILE, 'w') as f:
        f.write(f"Full Rating Analysis Report - {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis includes rating estimation for player: '{PLAYER_ENGINE_NAME}'\n")
        f.write("="*80 + "\n\n")

        plt.figure(figsize=(14, 8))

        for scenario in weight_scenarios:
            name = scenario['name']
            weights = scenario['weights']
            
            print(f"Processing scenario: {name}...")
            
            hit_percentages = calculate_hit_percentage(main_df, weights)
            
            # Prepare data for plotting (excluding the player engine)
            plot_data = pd.DataFrame(list(hit_percentages.items()), columns=['engine_name', 'hit_percentage'])
            plot_data['rating'] = plot_data['engine_name'].map(engine_ratings)
            plot_data.dropna(subset=['rating'], inplace=True) # This removes Desjardins373 for plotting
            plot_data.sort_values('rating', inplace=True)

            f.write(f"--- Scenario: {name} ---\n")
            f.write(f"Weights: {weights}\n\n")
            f.write(pd.DataFrame(list(hit_percentages.items()), columns=['Engine', 'Hit %']).round(2).to_string(index=False))
            f.write("\n\n")

            plt.scatter(plot_data['rating'], plot_data['hit_percentage'], label=f'{name} - Actual Engines')
            
            params = fit_elo_curve(plot_data['rating'], plot_data['hit_percentage'])
            if params is not None:
                x_fit = np.linspace(plot_data['rating'].min(), plot_data['rating'].max(), 200)
                y_fit = elo_curve(x_fit, *params)
                plt.plot(x_fit, y_fit, label=f'{name} - Fitted Curve')

                # --- RATING ESTIMATION ---
                if PLAYER_ENGINE_NAME in hit_percentages:
                    player_hit_percentage = hit_percentages[PLAYER_ENGINE_NAME]
                    estimated_rating = estimate_rating_from_percentage(player_hit_percentage, params)
                    
                    if estimated_rating:
                        f.write(f"--- Rating Estimation for {PLAYER_ENGINE_NAME} ---\n")
                        f.write(f"Hit Percentage: {player_hit_percentage:.2f}%\n")
                        f.write(f"ESTIMATED ELO RATING: {estimated_rating:.0f}\n")
                        # Plot the estimated point
                        plt.scatter([estimated_rating], [player_hit_percentage], c='red', s=100, zorder=5, 
                                    marker='*', label=f'{PLAYER_ENGINE_NAME} (Estimated)')
                    else:
                        f.write(f"Could not estimate rating for {PLAYER_ENGINE_NAME}.\n")
            
            f.write("\n" + "-"*80 + "\n\n")

        plt.title('Engine Hit Percentage vs. Elo Rating', fontsize=16)
        plt.xlabel('Engine Elo Rating', fontsize=12)
        plt.ylabel('Weighted Hit Percentage (%)', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(OUTPUT_PLOT_FILE)
        print(f"\nSuccessfully saved plot to '{OUTPUT_PLOT_FILE}'")

    end_time = datetime.now()
    print(f"Analysis report saved to '{OUTPUT_REPORT_FILE}'")
    print(f"--- Analysis Complete in {end_time - start_time} ---")


if __name__ == '__main__':
    main()