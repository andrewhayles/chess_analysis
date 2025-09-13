import json
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import warnings

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
        # The 'move_rank' column directly gives the rank (0, 1, 2, or -1).
        # We map these ranks to our weights. A rank of -1 gets a weight of 0.
        weighted_hits = group['move_rank'].apply(lambda rank: weights.get(rank, 0)).sum()

        # The total possible score is if the engine played the best move (rank 0) every time.
        # The weight for the top move (rank 0) is considered the maximum possible score per position.
        total_possible_score = len(group) * weights.get(0, 1) 
        
        engine_hit_counts[engine] = (weighted_hits / total_possible_score) * 100 if total_possible_score > 0 else 0
            
    return engine_hit_counts

# Load data
try:
    df = pd.read_csv('granular_analysis_log_top3.csv')
    with open('oracle_cache_top3.json', 'r') as f:
        oracle_data = json.load(f)
    # Load engine ratings from the CSV file
    engine_ratings_df = pd.read_csv('real_engines.csv')
    engine_ratings = pd.Series(engine_ratings_df.rating.values, index=engine_ratings_df.engine_name).to_dict()

except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    exit()

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

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Engine Data')
    x_fit = np.linspace(min(x_data) * 0.9, max(x_data) * 1.1, 200)
    y_fit = logistic_function(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r-', label=f'Logistic Fit (R²={r_squared:.4f})')
    plt.axhline(y=player_hp, color='g', linestyle='--', label=f'{player_name} Hit % ({player_hp:.2f}%)')
    
    estimated_rating = get_rating_from_percentage(player_hp, *popt)
    if not np.isnan(estimated_rating):
        plt.axvline(x=estimated_rating, color='purple', linestyle=':', label=f'Estimated Rating: {estimated_rating:.0f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Hit Count Percentage')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    return r_squared, estimated_rating

# --- 3. Optimization of Weights and Ratings ---

def objective_function_weights(weights_array, df, engine_ratings):
    """Objective function to maximize R^2 by adjusting weights."""
    weights = {0: weights_array[0], 1: weights_array[1], 2: weights_array[2]}
    engine_hit_percentages = calculate_hit_percentage(df, weights) 
    
    summary_df = pd.DataFrame(list(engine_hit_percentages.items()), columns=['engine_name', 'hit_percentage'])
    summary_df['rating'] = summary_df['engine_name'].map(engine_ratings)
    
    fit_df = summary_df.dropna(subset=['rating'])
    
    x_data = fit_df['rating'].values
    y_data = fit_df['hit_percentage'].values
    
    if len(x_data) < 3: return 1.0
    
    try:
        bounds = ([min(y_data), 0, min(x_data)], [max(y_data)+1, 0.1, max(x_data)])
        popt, _ = curve_fit(logistic_function, x_data, y_data, p0=[max(y_data), 0.005, np.median(x_data)], bounds=bounds, maxfev=10000)
        residuals = y_data - logistic_function(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return 1 - r_squared
    except (RuntimeError, ValueError):
        return 1.0

# Initial weights and optimization
initial_weights = [1.0, 0.5, 0.1]
print("--- Optimizing Weights ---")
result_weights = minimize(objective_function_weights, initial_weights, args=(df, engine_ratings), method='Nelder-Mead')
optimal_weights_array = result_weights.x
optimal_weights = {0: optimal_weights_array[0], 1: optimal_weights_array[1], 2: optimal_weights_array[2]}
print(f"Optimal Weights Found: {optimal_weights}")


# Recalculate with optimal weights
engine_hit_percentages = calculate_hit_percentage(df, optimal_weights)

summary_df = pd.DataFrame(list(engine_hit_percentages.items()), columns=['engine_name', 'hit_percentage'])
summary_df['rating'] = summary_df['engine_name'].map(engine_ratings)

# Separate player data from engine data for curve fitting
player_name = 'Desjardins373'
player_row = summary_df[summary_df['engine_name'] == player_name]
player_hit_percentage = player_row['hit_percentage'].iloc[0] if not player_row.empty else 0

fit_df = summary_df.dropna(subset=['rating'])
fit_df = fit_df[fit_df['engine_name'] != player_name].sort_values('rating').reset_index(drop=True)


# --- 4. Display Hit Percentage Table ---
print("\n--- Hit Percentage Analysis with Optimal Weights ---")
print(summary_df.sort_values('rating', na_position='last').reset_index(drop=True)[['engine_name', 'rating', 'hit_percentage']].to_string())
print("\n" + "="*40)
print(f"Your ({player_name}) Overall Hit Percentage: {player_hit_percentage:.2f}%")
print("="*40 + "\n")


# --- 5. Rating Adjustment and Re-analysis ---

def objective_function_ratings(ratings, hit_percentages):
    """Objective function to minimize (1 - R^2) by adjusting ratings."""
    try:
        bounds = ([min(hit_percentages), 0, min(ratings)], [max(hit_percentages)+1, 0.1, max(ratings)])
        popt, _ = curve_fit(logistic_function, ratings, hit_percentages, p0=[max(hit_percentages), 0.005, np.median(ratings)], bounds=bounds, maxfev=10000)
        residuals = hit_percentages - logistic_function(ratings, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((hit_percentages - np.mean(hit_percentages))**2)
        if ss_tot == 0: return 1.0
        r_squared = 1 - (ss_res / ss_tot)
        return 1 - r_squared
    except (RuntimeError, ValueError):
        return 1.0

x_data = fit_df['rating'].values
y_data = fit_df['hit_percentage'].values
r_squared_initial = 0.0

print("--- Logistic Regression Analysis ---")
try:
    bounds = ([min(y_data), 0, min(x_data)], [max(y_data)+1, 0.1, max(x_data)])
    popt, _ = curve_fit(logistic_function, x_data, y_data, p0=[max(y_data), 0.005, np.median(x_data)], bounds=bounds, maxfev=10000)
    r_squared_initial, initial_rating = plot_results(x_data, y_data, popt, player_hit_percentage,
                                                     'Initial Hit Count % vs. Engine Rating',
                                                     'Engine Rating', 'initial_rating_correlation.png', player_name)

    initial_rating_str = f"{initial_rating:.2f}" if not np.isnan(initial_rating) else "N/A"
    print(f"Initial R-squared: {r_squared_initial:.4f}")
    print(f"Initial Estimated Rating: {initial_rating_str}")

except (RuntimeError, ValueError) as e:
    print(f"Could not perform initial logistic fit: {e}")
    initial_rating = np.nan

if r_squared_initial < 0.9 and len(x_data) > 0:
    print("\nInitial correlation is below 0.9. Adjusting ratings...")
    initial_ratings = fit_df['rating'].values
    rating_bounds = [(r * 0.5, r * 1.5) for r in initial_ratings]

    result = minimize(objective_function_ratings, initial_ratings, args=(y_data,), method='L-BFGS-B', bounds=rating_bounds)
    adjusted_ratings = result.x

    fit_df['adjusted_rating'] = adjusted_ratings
    print("\nAdjusted Engine Ratings:")
    print(fit_df[['engine_name', 'rating', 'adjusted_rating']].round(0).to_string())

    try:
        x_data_adj = fit_df['adjusted_rating'].values
        popt_adj, _ = curve_fit(logistic_function, x_data_adj, y_data, p0=[max(y_data), 0.005, np.median(x_data_adj)], bounds=bounds, maxfev=10000)
        r_squared_adj, final_rating = plot_results(x_data_adj, y_data, popt_adj, player_hit_percentage,
                                                   'Adjusted Hit Count % vs. Engine Rating',
                                                   'Adjusted Engine Rating', 'adjusted_rating_correlation.png', player_name)

        final_rating_str = f"{final_rating:.2f}" if not np.isnan(final_rating) else "N/A"
        print(f"\nAdjusted R-squared: {r_squared_adj:.4f}")
        print(f"Final Estimated Rating (with adjusted data): {final_rating_str}")
    except (RuntimeError, ValueError) as e:
        print(f"Could not perform adjusted logistic fit: {e}")
else:
    print("\nInitial correlation is already above 0.9. No adjustment needed.")
