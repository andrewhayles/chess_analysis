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
OUTPUT_REPORT_FILE = 'full_rating_analysis_report.txt'
OUTPUT_PLOT_FILE = 'full_rating_analysis_plot.png'

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
    
    # Create a dictionary for engine ratings for easy lookup
    engine_ratings = pd.Series(engines_df.rating.values, index=engines_df.engine_name).to_dict()
    
    # Add a 'rating' column to the main dataframe
    main_df['rating'] = main_df['engine_name'].map(engine_ratings)
    
    # Drop any rows where the engine rating could not be found
    main_df.dropna(subset=['rating'], inplace=True)
    
    print(f"Successfully loaded and merged data: {len(main_df)} records found.")
    return main_df, engine_ratings

# --- 2. Analysis and Modeling ---

def calculate_hit_percentage(df, weights):
    """Calculates the weighted hit percentage for each engine."""
    engine_scores = {}
    for engine, group in df.groupby('engine_name'):
        weighted_hits = group['move_rank'].apply(lambda rank: weights.get(rank, 0)).sum()
        # The maximum possible score is the number of games * the highest possible weight (for move_rank 0)
        total_possible_score = len(group) * weights.get(0, 1) 
        engine_scores[engine] = (weighted_hits / total_possible_score) * 100 if total_possible_score > 0 else 0
    return engine_scores

def elo_curve(x, a, b, c):
    """Defines the logistic curve used to model Elo ratings."""
    return a / (1 + np.exp(-b * (x - c)))

def fit_elo_curve(ratings, hit_percentages):
    """Fits the Elo curve to the data and returns the curve parameters."""
    # Provide reasonable initial guesses for the parameters [a, b, c]
    # a: Maximum hit rate (e.g., 100%)
    # b: Steepness of the curve
    # c: The rating at which the curve is steepest (midpoint)
    initial_guesses = [100.0, 0.01, np.median(ratings)]
    
    try:
        params, _ = curve_fit(elo_curve, ratings, hit_percentages, p0=initial_guesses, maxfev=5000)
        return params
    except RuntimeError:
        print("  Warning: Could not find optimal parameters for the Elo curve.")
        return None

# --- 3. Main Execution ---

def main():
    """Main function to run the full analysis and generate reports."""
    start_time = datetime.now()
    print("--- Starting Full Rating Analysis ---")
    
    main_df, engine_ratings = load_data(GRANULAR_LOG_FILE, ENGINES_CSV_FILE)
    if main_df is None:
        return # Stop execution if data loading failed

    # --- Define Weight Scenarios ---
    weight_scenarios = [
        {'name': 'Top1_Only', 'weights': {0: 1.0, 1: 0.0, 2: 0.0}},
        {'name': 'Balanced', 'weights': {0: 1.0, 1: 0.5, 2: 0.25}},
        {'name': 'Linear_Decay', 'weights': {0: 1.0, 1: 0.66, 2: 0.33}},
    ]
    
    # Open the report file to save all text output
    with open(OUTPUT_REPORT_FILE, 'w') as f:
        f.write(f"Full Rating Analysis Report - {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
        f.write(f"Analyzed data from: '{GRANULAR_LOG_FILE}'\n")
        f.write(f"Total records processed: {len(main_df)}\n")
        f.write(f"Number of unique engines: {main_df['engine_name'].nunique()}\n")
        f.write("="*80 + "\n\n")

        plt.figure(figsize=(14, 8))

        for scenario in weight_scenarios:
            name = scenario['name']
            weights = scenario['weights']
            
            print(f"Processing scenario: {name}...")
            
            hit_percentages = calculate_hit_percentage(main_df, weights)
            
            # Prepare data for plotting and curve fitting
            plot_data = pd.DataFrame(list(hit_percentages.items()), columns=['engine_name', 'hit_percentage'])
            plot_data['rating'] = plot_data['engine_name'].map(engine_ratings)
            plot_data.dropna(subset=['rating'], inplace=True)
            plot_data.sort_values('rating', inplace=True)

            # --- Reporting ---
            f.write(f"--- Scenario: {name} ---\n")
            f.write(f"Weights: {weights}\n\n")
            f.write(plot_data[['engine_name', 'rating', 'hit_percentage']].round(2).to_string(index=False))
            f.write("\n\n" + "-"*80 + "\n\n")

            # --- Plotting ---
            plt.scatter(plot_data['rating'], plot_data['hit_percentage'], label=f'{name} - Actual')
            
            # --- Curve Fitting ---
            params = fit_elo_curve(plot_data['rating'], plot_data['hit_percentage'])
            if params is not None:
                x_fit = np.linspace(plot_data['rating'].min(), plot_data['rating'].max(), 200)
                y_fit = elo_curve(x_fit, *params)
                plt.plot(x_fit, y_fit, label=f'{name} - Fit')

        # --- Finalize Plot ---
        plt.title('Engine Hit Percentage vs. Elo Rating (All Positions)', fontsize=16)
        plt.xlabel('Engine Elo Rating', fontsize=12)
        plt.ylabel('Weighted Hit Percentage (%)', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        
        try:
            plt.savefig(OUTPUT_PLOT_FILE)
            print(f"\nSuccessfully saved plot to '{OUTPUT_PLOT_FILE}'")
        except IOError as e:
            print(f"Error saving plot: {e}")

    end_time = datetime.now()
    print(f"Analysis report saved to '{OUTPUT_REPORT_FILE}'")
    print(f"--- Analysis Complete in {end_time - start_time} ---")


if __name__ == '__main__':
    main()