import json
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import os

# Suppress warnings that are handled in the code
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- 1. Load and Preprocess Data ---

def calculate_hit_percentage(df, weights):
    """
    Calculates the weighted hit percentage for each engine based on move_rank.
    """
    engine_hit_counts = {}
    for engine, group in df.groupby('engine_name'):
        # Map move ranks to weights. A rank of -1 (or any other not in weights) gets a weight of 0.
        weighted_hits = group['move_rank'].apply(lambda rank: weights.get(rank, 0)).sum()

        # The total possible score is if the engine played the best move (rank 0) every time.
        total_possible_score = len(group) * weights.get(0, 1) 
        
        engine_hit_counts[engine] = (weighted_hits / total_possible_score) * 100 if total_possible_score > 0 else 0
            
    return engine_hit_counts

def calculate_raw_hit_counts(df):
    """
    Calculates the raw, unweighted hit counts for the top 3 moves for each engine.
    """
    raw_counts = {}
    for engine, group in df.groupby('engine_name'):
        total_moves = len(group)
        if total_moves == 0:
            continue
        
        top1_hits = (group['move_rank'] == 0).sum()
        top2_hits = (group['move_rank'] == 1).sum()
        top3_hits = (group['move_rank'] == 2).sum()
        
        raw_counts[engine] = {
            'Total Moves': total_moves,
            'Top 1 Hits': top1_hits,
            'Top 2 Hits': top2_hits,
            'Top 3 Hits': top3_hits,
            'Top 1 %': (top1_hits / total_moves) * 100,
            'Top 2 %': (top2_hits / total_moves) * 100,
            'Top 3 %': (top3_hits / total_moves) * 100,
        }
    return raw_counts

# --- 2. Logistic Curve Analysis and Visualization ---

def logistic_function(x, L, k, x0):
    """Logistic function for fitting."""
    return L / (1 + np.exp(-k * (x - x0)))

def get_rating_from_percentage(p, L, k, x0):
    """Inverse of the logistic function to estimate rating."""
    if p <= 0 or p >= L:
        return np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = L / p - 1
    if arg <= 0 or np.isinf(arg) or np.isnan(arg):
        return np.nan
    return x0 - (np.log(arg) / k)

def plot_results(x_data, y_data, popt, player_hp, title, xlabel, filename, player_name='Player'):
    """Helper function to plot the results."""
    residuals = y_data - logistic_function(x_data, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    plt.figure(figsize=(12, 7))
    plt.scatter(x_data, y_data, label='Engine Data', color='blue', zorder=5)
    
    x_range = max(x_data) - min(x_data) if len(x_data) > 1 else 1000
    x_fit = np.linspace(min(x_data) - 0.1 * x_range, max(x_data) + 0.1 * x_range, 400)
    y_fit = logistic_function(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r-', label=f'Logistic Fit (R²={r_squared:.4f})', linewidth=2)
    
    plt.axhline(y=player_hp, color='green', linestyle='--', label=f'{player_name} Hit % ({player_hp:.2f}%)')
    
    estimated_rating = get_rating_from_percentage(player_hp, *popt)
    if not np.isnan(estimated_rating):
        plt.axvline(x=estimated_rating, color='purple', linestyle=':', label=f'Estimated Rating: {estimated_rating:.0f}')

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Hit Count Percentage', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return r_squared, estimated_rating

# --- 3. Main Analysis Function ---

def run_analysis(df, engine_ratings, weights_config):
    """
    Runs the full analysis for a given weight configuration.
    """
    weights = weights_config['weights']
    scenario_name = weights_config['name']
    
    print("\n" + "="*80)
    print(f"--- RUNNING ANALYSIS FOR SCENARIO: {scenario_name} ---")
    print(f"--- Weights: {weights} ---")
    print("="*80 + "\n")

    # Recalculate with current weights for ALL engines for reporting
    engine_hit_percentages = calculate_hit_percentage(df, weights)
    summary_df = pd.DataFrame(list(engine_hit_percentages.items()), columns=['engine_name', 'hit_percentage'])
    summary_df['rating'] = summary_df['engine_name'].map(engine_ratings)

    # Separate player data from engine data for curve fitting
    player_name = 'Desjardins373'
    player_row = summary_df[summary_df['engine_name'] == player_name]
    player_hit_percentage = player_row['hit_percentage'].iloc[0] if not player_row.empty else 0

    # Create the final fit_df for analysis using only Stockfish engines
    fit_df = summary_df.dropna(subset=['rating'])
    fit_df = fit_df[~fit_df['engine_name'].str.contains('maia', case=False)]
    fit_df = fit_df[fit_df['engine_name'] != player_name].sort_values('rating').reset_index(drop=True)

    # --- Rating Adjustment and Re-analysis ---
    def objective_function_ratings_split(adjustable_ratings, adjustable_hits, fixed_ratings, fixed_hits):
        """Objective function that combines fixed and adjustable ratings for R^2 calculation."""
        all_ratings = np.concatenate([adjustable_ratings, fixed_ratings])
        all_hits = np.concatenate([adjustable_hits, fixed_hits])
        try:
            p0 = [max(all_hits), 0.005, np.median(all_ratings)]
            bounds = ([min(all_hits)-1, 0, min(all_ratings)-1], [max(all_hits)*1.1, 0.1, max(all_ratings)*1.1])
            popt, _ = curve_fit(logistic_function, all_ratings, all_hits, p0=p0, bounds=bounds, maxfev=10000)
            residuals = all_hits - logistic_function(all_ratings, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((all_hits - np.mean(all_hits))**2)
            if ss_tot == 0: return 1.0
            r_squared = 1 - (ss_res / ss_tot)
            return 1 - r_squared
        except (RuntimeError, ValueError):
            return 1.0

    x_data = fit_df['rating'].values
    y_data = fit_df['hit_percentage'].values
    r_squared_initial, initial_rating = (0.0, np.nan)

    print("--- Logistic Regression Analysis (Stockfish engines) ---")
    if len(x_data) > 2:
        try:
            p0 = [max(y_data), 0.005, np.median(x_data)]
            bounds = ([min(y_data)-1, 0, min(x_data)-1], [max(y_data)*1.1, 0.1, max(x_data)*1.1])
            popt, _ = curve_fit(logistic_function, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
            r_squared_initial, initial_rating = plot_results(x_data, y_data, popt, player_hit_percentage,
                                                             f'Initial Hit Count % vs. Engine Rating ({scenario_name})',
                                                             'Engine Rating', f'initial_rating_correlation_{scenario_name}.png', player_name)
            initial_rating_str = f"{initial_rating:.2f}" if not np.isnan(initial_rating) else "N/A"
            print(f"Initial R-squared: {r_squared_initial:.4f}")
            print(f"Initial Estimated Rating: {initial_rating_str}")
        except (RuntimeError, ValueError) as e:
            print(f"Could not perform initial logistic fit: {e}")
    else:
        print("Not enough engines to perform logistic fit.")

    final_summary_df = summary_df.copy()
    final_summary_df['adjusted_rating'] = final_summary_df['rating']
    r_squared_adj, final_rating = r_squared_initial, initial_rating

    if r_squared_initial < 0.9 and len(x_data) > 0:
        print("\nInitial correlation is below 0.9. Adjusting ratings for node-based engines...")
        
        adjustable_df = fit_df[fit_df['engine_name'].str.contains('stockfish', case=False) & ~fit_df['engine_name'].str.contains('full', case=False)].copy()
        fixed_df = fit_df[~fit_df['engine_name'].isin(adjustable_df['engine_name'])].copy()

        if not adjustable_df.empty:
            initial_adjustable_ratings = adjustable_df['rating'].values
            adjustable_hits = adjustable_df['hit_percentage'].values
            fixed_ratings = fixed_df['rating'].values
            fixed_hits = fixed_df['hit_percentage'].values
            
            ORACLE_RATING_CAP = 3650
            rating_bounds = [(r * 0.5, min(r * 1.5, ORACLE_RATING_CAP)) for r in initial_adjustable_ratings]

            result = minimize(objective_function_ratings_split, initial_adjustable_ratings, 
                              args=(adjustable_hits, fixed_ratings, fixed_hits), 
                              method='L-BFGS-B', bounds=rating_bounds)
            adjusted_sf_ratings = result.x

            adjusted_map = dict(zip(adjustable_df['engine_name'], adjusted_sf_ratings))
            final_summary_df['adjusted_rating'] = final_summary_df['engine_name'].map(adjusted_map).fillna(final_summary_df['rating'])

            final_fit_df = final_summary_df.dropna(subset=['rating'])
            final_fit_df = final_fit_df[final_fit_df['engine_name'] != player_name].sort_values('rating').reset_index(drop=True)

            print("\nAdjusted Engine Ratings (only node-based engines were modified):")
            print(final_fit_df[~final_fit_df['engine_name'].str.contains('maia', case=False)][['engine_name', 'rating', 'adjusted_rating']].round(0).to_string())

            try:
                x_data_adj = final_fit_df[~final_fit_df['engine_name'].str.contains('maia', case=False)]['adjusted_rating'].values
                y_data_adj = final_fit_df[~final_fit_df['engine_name'].str.contains('maia', case=False)]['hit_percentage'].values
                
                p0_adj = [max(y_data_adj), 0.005, np.median(x_data_adj)]
                bounds_adj = ([min(y_data_adj)-1, 0, min(x_data_adj)-1], [max(y_data_adj)*1.1, 0.1, max(x_data_adj)*1.1])
                popt_adj, _ = curve_fit(logistic_function, x_data_adj, y_data_adj, p0=p0_adj, bounds=bounds_adj, maxfev=10000)
                r_squared_adj, final_rating = plot_results(x_data_adj, y_data_adj, popt_adj, player_hit_percentage,
                                                           f'Adjusted Hit Count % vs. Adjusted Rating ({scenario_name})',
                                                           'Adjusted Engine Rating', f'adjusted_rating_correlation_{scenario_name}.png', player_name)
                final_rating_str = f"{final_rating:.2f}" if not np.isnan(final_rating) else "N/A"
                print(f"\nAdjusted R-squared: {r_squared_adj:.4f}")
                print(f"Final Estimated Rating (with adjusted data): {final_rating_str}")
            except (RuntimeError, ValueError) as e:
                print(f"Could not perform adjusted logistic fit: {e}")
        else:
            print("\nNo node-based engines found to adjust.")
    else:
        print("\nInitial correlation is already above 0.9 or not enough data. No adjustment needed.")

    # --- Save Results to File ---
    try:
        with open(f'rating_analysis_results_{scenario_name}.txt', 'w') as f:
            f.write(f"Rating Analysis Results for Scenario: {scenario_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Weights Used for Analysis: {weights}\n\n")
            
            f.write("="*50 + "\n")
            f.write("Final Engine Data:\n")
            f.write("="*50 + "\n")
            f.write(final_summary_df.sort_values('rating', na_position='last')
                    [['engine_name', 'rating', 'adjusted_rating', 'hit_percentage']]
                    .round({'rating': 0, 'adjusted_rating': 0, 'hit_percentage': 2}).to_string(index=False))
            f.write("\n\n")

            f.write("="*50 + "\n")
            f.write("Analysis Summary (based on Stockfish engines):\n")
            f.write("="*50 + "\n")
            f.write(f"Your ({player_name}) Hit Percentage: {player_hit_percentage:.2f}%\n\n")
            
            f.write(f"Initial R-squared (before rating adjustment): {r_squared_initial:.4f}\n")
            initial_rating_str = f"{initial_rating:.0f}" if not np.isnan(initial_rating) else "N/A"
            f.write(f"Initial Estimated Rating: {initial_rating_str}\n\n")
            
            f.write(f"Adjusted R-squared (after rating adjustment): {r_squared_adj:.4f}\n")
            final_rating_str = f"{final_rating:.0f}" if not np.isnan(final_rating) else "N/A"
            f.write(f"Final Estimated Rating: {final_rating_str}\n")
        print(f"Results for {scenario_name} successfully saved to rating_analysis_results_{scenario_name}.txt")
    except IOError as e:
        print(f"Error saving results to file for {scenario_name}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Load data once
    try:
        main_df = pd.read_csv('granular_analysis_log_top3.csv')
        main_engine_ratings = pd.Series(pd.read_csv('real_engines.csv').set_index('engine_name')['rating']).to_dict()
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        exit()

    # Save raw hit counts once
    try:
        raw_counts_data = calculate_raw_hit_counts(main_df)
        raw_counts_df = pd.DataFrame.from_dict(raw_counts_data, orient='index')
        raw_counts_df.index.name = 'engine_name'
        raw_counts_df.reset_index(inplace=True)
        ratings_for_sort_df = pd.DataFrame(list(main_engine_ratings.items()), columns=['engine_name', 'rating'])
        raw_counts_df = pd.merge(raw_counts_df, ratings_for_sort_df, on='engine_name', how='left')
        
        with open('raw_hit_counts.txt', 'w') as f:
            f.write(f"Raw Hit Count Data - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            f.write(raw_counts_df.sort_values('rating', na_position='last').drop(columns='rating')
                    .round(2).to_string(index=False))
        print("Raw hit counts successfully saved to raw_hit_counts.txt")
    except IOError as e:
        print(f"Error saving raw hit counts to file: {e}")

    # --- Define Weight Scenarios ---
    weight_scenarios = [
        {'name': 'Top1_Only', 'weights': {0: 1.0, 1: 0.0, 2: 0.0}},
        {'name': 'Balanced', 'weights': {0: 1.0, 1: 0.5, 2: 0.1}},
        {'name': 'Front_Heavy', 'weights': {0: 1.0, 1: 0.2, 2: 0.1}},
        {'name': 'Top1_Focus', 'weights': {0: 1.0, 1: 0.1, 2: 0.01}}
    ]

    # --- Run Analysis for Each Scenario ---
    for scenario in weight_scenarios:
        run_analysis(main_df, main_engine_ratings, scenario)