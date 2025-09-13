import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ==============================================================================
# --- Configuration ---
# ==============================================================================
GRANULAR_LOG_FILE = "granular_analysis_log.csv"
ENGINES_CSV_PATH = "real_engines.csv"
OUTPUT_PLOT_FILE = "engine_match_rate_vs_rating_logistic.png"

# ==============================================================================
# --- Analysis and Plotting ---
# ==============================================================================

def logistic_function(x, L, k, x0):
    """
    Represents the logistic (sigmoid) function.
    L: The curve's maximum value (max match rate, ~1.0).
    k: The logistic growth rate or steepness of the curve.
    x0: The x-value (Elo) of the sigmoid's midpoint.
    """
    try:
        return L / (1 + np.exp(-k * (x - x0)))
    except OverflowError:
        return np.inf

def plot_logistic_fit_rate_vs_rating():
    """
    Loads analysis results, fits a logistic curve to map Elo to match rate,
    and plots the data.
    """
    print("--- Starting Logistic Curve Fitting (Rate vs. Rating) Script ---")

    # --- 1. Load and Aggregate Data ---
    try:
        log_df = pd.read_csv(GRANULAR_LOG_FILE)
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find a required file: {e}. Make sure '{GRANULAR_LOG_FILE}' and '{ENGINES_CSV_PATH}' are in the same directory.")
        return

    # Calculate the oracle match rate for each engine
    match_rates = log_df.groupby('engine_name')['score'].mean().reset_index()
    match_rates.rename(columns={'score': 'match_rate'}, inplace=True)
    print("\n[INFO] Calculated Match Rates:")
    print(match_rates)

    # --- 2. Merge Data and Prepare for Fitting ---
    merged_df = pd.merge(match_rates, engines_df[['engine_name', 'rating']], on='engine_name', how='left')
    fit_data = merged_df.dropna(subset=['rating'])
    
    if len(fit_data) < 3:
        print("\n[ERROR] Need at least 3 engines with known ratings to fit a curve.")
        return

    # *** AXES SWAPPED: x is now rating, y is now match_rate ***
    x_data = fit_data['rating']
    y_data = fit_data['match_rate']

    # --- 3. Fit the Logistic Curve ---
    print("\n[INFO] Fitting logistic curve to the data...")
    try:
        # Adjust initial guesses and bounds for the new axis orientation
        # L (max rate) is ~1.0; k (steepness) is small; x0 (midpoint) is a rating
        initial_guesses = [1.0, 0.005, np.median(x_data)]
        bounds = ([0.5, 0.0001, 500], [1.5, 0.1, 4000])
        
        params, covariance = curve_fit(logistic_function, x_data, y_data, p0=initial_guesses, bounds=bounds, maxfev=5000)
        L_fit, k_fit, x0_fit = params
        print(f"[INFO] Curve fitted with parameters: L={L_fit:.2f}, k={k_fit:.4f}, x0={x0_fit:.2f}")

        y_pred = logistic_function(x_data, L_fit, k_fit, x0_fit)
        r2 = r2_score(y_data, y_pred)
        print(f"[INFO] R-squared of the fit: {r2:.4f}")

    except RuntimeError as e:
        print(f"\n[ERROR] Could not fit the curve to the data: {e}")
        return

    # --- 4. Plot the Results ---
    print(f"\n[INFO] Generating plot and saving to '{OUTPUT_PLOT_FILE}'...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot of the actual engine data
    ax.scatter(x_data, y_data, label='Known Engine Ratings', color='royalblue', zorder=5)

    # Annotate each point with the engine name
    for i, txt in enumerate(fit_data['engine_name']):
        ax.annotate(txt, (x_data.iloc[i], y_data.iloc[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    # Plot the fitted logistic curve
    x_curve = np.linspace(min(x_data) - 100, max(x_data) + 100, 400)
    y_curve = logistic_function(x_curve, L_fit, k_fit, x0_fit)
    ax.plot(x_curve, y_curve, label=f'Fitted Logistic Curve (R²={r2:.3f})', color='darkorange', linewidth=2)

    # --- 5. Formatting ---
    ax.set_title('Engine Match Rate vs. Elo Rating', fontsize=16)
    ax.set_xlabel('Known Elo Rating', fontsize=12)
    ax.set_ylabel('Oracle Match Rate', fontsize=12)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILE)
    plt.show()

    print("\n[DONE] Plotting complete.")


if __name__ == "__main__":
    plot_logistic_fit_rate_vs_rating()
