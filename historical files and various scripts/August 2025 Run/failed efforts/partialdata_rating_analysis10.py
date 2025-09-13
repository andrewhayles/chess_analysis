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
        weighted_hits = group['move_rank'].apply(lambda rank: weights.get(rank, 0)).sum()
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
        if total_moves == 0: continue
        
        top1_hits = (group['move_rank'] == 0).sum()
        top2_hits = (group['move_rank'] == 1).sum()
        top3_hits = (group['move_rank'] == 2).sum()
        
        raw_counts[engine] = {
            'Total Moves': total_moves, 'Top 1 Hits': top1_hits, 'Top 2 Hits': top2_hits,
            'Top 3 Hits': top3_hits, 'Top 1 %': (top1_hits / total_moves) * 100,
            'Top 2 %': (top2_hits / total_moves) * 100, 'Top 3 %': (top3_hits / total_moves) * 100,
        }
    return raw_counts

# --- 2. Logistic Curve Analysis and Visualization ---

def logistic_function(x, L, k, x0):
    """Logistic function for fitting."""
    return L / (1 + np.exp(-k * (x - x0)))

def get_rating_from_percentage(p, L, k, x0):
    """Inverse of the logistic function to estimate rating."""
    if p <= 0 or p >= L: return np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        arg = L / p - 1
    if arg <= 0 or np.isinf(arg) or np.isnan(arg): return np.nan
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
    """Runs the full analysis for a given weight configuration."""
    weights = weights_config['weights']
    scenario_name = weights_config['name']
    
    print(f"\n--- RUNNING ANALYSIS FOR SCENARIO: {scenario_name} ---")

    engine_hit_percentages = calculate_hit_percentage(df, weights)
    summary_df = pd.DataFrame(list(engine_hit_percentages.items()), columns=['engine_name', 'hit_percentage'])
    summary_df['rating'] = summary_df['engine_name'].map(engine_ratings)

    player_name = 'Desjardins373'
    player_row = summary_df[summary_df['engine_name'] == player_name]
    player_hit_percentage = player_row['hit_percentage'].iloc[0] if not player_row.empty else 0

    fit_df = summary_df.dropna(subset=['rating'])
    fit_df = fit_df[~fit_df['engine_name'].str.contains('maia', case=False)]
    fit_df = fit_df[fit_df['engine_name'] != player_name].sort_values('rating').reset_index(drop=True)

    x_data = fit_df['rating'].values
    y_data = fit_df['hit_percentage'].values
    r_squared, rating_estimate = (0.0, np.nan)

    if len(x_data) > 2:
        try:
            p0 = [max(y_data), 0.005, np.median(x_data)]
            bounds = ([min(y_data)-1, 0, min(x_data)-1], [max(y_data)*1.1, 0.1, max(x_data)*1.1])
            popt, _ = curve_fit(logistic_function, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
            r_squared, rating_estimate = plot_results(x_data, y_data, popt, player_hit_percentage,
                                                     f'Hit Count % vs. Engine Rating ({scenario_name})',
                                                     'Engine Rating', f'final_rating_correlation_{scenario_name}.png', player_name)
            print(f"  R-squared: {r_squared:.4f}")
            print(f"  Estimated Rating: {rating_estimate:.0f}" if not np.isnan(rating_estimate) else "N/A")
        except (RuntimeError, ValueError) as e:
            print(f"  Could not perform logistic fit: {e}")
    else:
        print("  Not enough engines to perform logistic fit.")

    try:
        with open(f'rating_analysis_results_{scenario_name}.txt', 'w') as f:
            f.write(f"Rating Analysis Results for Scenario: {scenario_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Weights Used: {weights}\n\n")
            f.write(summary_df.sort_values('rating', na_position='last').round(2).to_string(index=False))
            f.write("\n\n" + "="*50 + "\n")
            f.write("Analysis Summary:\n")
            f.write(f"  R-squared: {r_squared:.4f}\n")
            f.write(f"  Your ({player_name}) Hit Percentage: {player_hit_percentage:.2f}%\n")
            rating_str = f"{rating_estimate:.0f}" if not np.isnan(rating_estimate) else "N/A"
            f.write(f"  Final Estimated Rating: {rating_str}\n")
        print(f"  Results for {scenario_name} saved.")
    except IOError as e:
        print(f"  Error saving results to file for {scenario_name}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # --- MODIFIED: Load and combine both granular data files ---
    DATA_FILE_OLD = 'granular_analysis_log_top3.csv'
    DATA_FILE_NEW = 'granular_analysis_log_updated.csv'
    
    df_list = []
    if os.path.exists(DATA_FILE_OLD):
        print(f"Loading old data from '{DATA_FILE_OLD}'...")
        df_list.append(pd.read_csv(DATA_FILE_OLD))
    
    if os.path.exists(DATA_FILE_NEW):
        print(f"Loading new data from '{DATA_FILE_NEW}'...")
        df_list.append(pd.read_csv(DATA_FILE_NEW))

    if not df_list:
        print("FATAL ERROR: No data files found. Exiting.")
        exit()

    print("Combining data files for a complete analysis...")
    main_df = pd.concat(df_list, ignore_index=True).drop_duplicates()
    print(f"Total unique analysis entries: {len(main_df)}")
    # --- End of Modification ---

    try:
        engine_ratings_file = 'real_engines.csv'
        print(f"Loading engine ratings from '{engine_ratings_file}'...")
        main_engine_ratings = pd.Series(pd.read_csv(engine_ratings_file).set_index('engine_name')['rating']).to_dict()
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Engine ratings file not found: {e}")
        exit()

    # Save raw hit counts once using the combined data
    try:
        raw_counts_df = pd.DataFrame.from_dict(calculate_raw_hit_counts(main_df), orient='index')
        raw_counts_df.index.name = 'engine_name'
        raw_counts_df = raw_counts_df.reset_index().merge(
            pd.DataFrame(list(main_engine_ratings.items()), columns=['engine_name', 'rating']),
            on='engine_name', how='left'
        )
        with open('raw_hit_counts_final.txt', 'w') as f:
            f.write(f"Raw Hit Count Data (Combined) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            f.write(raw_counts_df.sort_values('rating', na_position='last').drop(columns='rating').round(2).to_string(index=False))
        print("Raw hit counts successfully saved to raw_hit_counts_final.txt")
    except IOError as e:
        print(f"Error saving raw hit counts: {e}")

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

    print("\n--- All analyses complete. ---")