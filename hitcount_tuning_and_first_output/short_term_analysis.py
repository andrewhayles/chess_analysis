import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Set the paths to your input files and desired output files.
LOG_FILE_PATH = 'rating_estimation_log.csv'
ENGINES_CSV_PATH = 'real_engines.csv'
OUTPUT_REPORT_PATH = 'analysis_report.txt'
OUTPUT_GRAPH_PATH = 'analysis_graph.png'

def analyze_rating_log():
    """
    Analyzes the rating estimation log to evaluate engine performance,
    calculate correlation between performance and rating, and generate a report and graph.
    """
    print("--- Starting Analysis of Rating Estimation Log ---")

    # --- 1. Load Data ---
    # Check if the necessary files exist before proceeding.
    if not os.path.exists(LOG_FILE_PATH):
        print(f"Error: Log file not found at '{LOG_FILE_PATH}'.")
        return
    if not os.path.exists(ENGINES_CSV_PATH):
        print(f"Error: Engines CSV file not found at '{ENGINES_CSV_PATH}'.")
        return

    try:
        # Load the log data and the engine reference data.
        log_df = pd.read_csv(LOG_FILE_PATH)
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except pd.errors.EmptyDataError:
        print(f"Error: '{LOG_FILE_PATH}' is empty. No data to analyze.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV files: {e}")
        return

    print(f"Successfully loaded {len(log_df)} records from '{LOG_FILE_PATH}'.")
    print(f"Successfully loaded {len(engines_df)} engine profiles from '{ENGINES_CSV_PATH}'.")

    # --- 2. Process and Aggregate Data ---
    # Calculate the average score and the number of positions tested for each engine.
    engine_performance = log_df.groupby('engine_name').agg(
        average_hit_score=('score', 'mean'),
        positions_analyzed=('fen', 'count')
    ).reset_index()

    # Merge the performance data with the engine ratings.
    analysis_df = pd.merge(engines_df, engine_performance, on='engine_name', how='inner')

    if len(analysis_df) < 2:
        print("\nInsufficient Data: The log contains data for fewer than two benchmark engines.")
        print("Cannot perform a correlation analysis. Please run the estimation script for more engines.")
        # Still, we can print the performance for the one engine we have.
        if not analysis_df.empty:
            print("\n--- Current Engine Performance ---")
            print(analysis_df[['engine_name', 'rating', 'average_hit_score', 'positions_analyzed']].to_string(index=False))
        return

    print("\n--- Current Engine Performance ---")
    print(analysis_df[['engine_name', 'rating', 'average_hit_score', 'positions_analyzed']].to_string(index=False))

    # --- 3. Perform Correlation Analysis ---
    # Use linear regression to find the relationship between hit score and rating.
    slope, intercept, r_value, p_value, std_err = linregress(analysis_df['average_hit_score'], analysis_df['rating'])
    r_squared = r_value ** 2

    print("\n--- Correlation Analysis ---")
    print(f"R-squared (Correlation of Performance vs. Rating): {r_squared:.4f}")
    print(f"This value indicates how well the 'average_hit_score' predicts the engine 'rating'.")
    print(f"A value closer to 1.0 suggests a strong linear relationship.")

    # --- 4. Generate Text Report ---
    try:
        with open(OUTPUT_REPORT_PATH, 'w') as f:
            f.write("--- Rating Estimation Analysis Report ---\n\n")
            f.write("This report summarizes the current state of the engine evaluation.\n\n")

            f.write("=== Engine Performance ===\n")
            f.write("Performance of each engine evaluated so far:\n")
            f.write(analysis_df[['engine_name', 'rating', 'average_hit_score', 'positions_analyzed']].to_string(index=False))
            f.write("\n\n")

            f.write("=== Correlation Analysis ===\n")
            f.write(f"R-squared: {r_squared:.4f}\n")
            f.write(f"Regression Formula: Estimated Rating = ({slope:.2f} * AverageHitScore) + {intercept:.2f}\n\n")
            f.write("The R-squared value measures how much of the variance in engine ratings\n")
            f.write("is explained by the average hit score. A higher value means a better fit.\n")

        print(f"\nSuccessfully generated text report: '{OUTPUT_REPORT_PATH}'")
    except Exception as e:
        print(f"\nError writing report file: {e}")


    # --- 5. Generate Visualization ---
    try:
        plt.figure(figsize=(12, 8))
        # Create a regression plot to visualize the relationship.
        sns.regplot(
            x='rating',
            y='average_hit_score',
            data=analysis_df,
            ci=None, # Disable confidence interval for a cleaner look
            line_kws={'color': 'red', 'linestyle': '--', 'label': f'Trendline (R² = {r_squared:.4f})'}
        )
        # Overlay scatter plot to clearly see individual engine points.
        sns.scatterplot(
            x='rating',
            y='average_hit_score',
            data=analysis_df,
            hue='engine_name',
            s=100, # Point size
            style='engine_name', # Different marker for each engine
            markers=True,
            palette='viridis'
        )

        # Add labels and titles for clarity.
        plt.title('Engine Performance (Average Hit Score) vs. Rating', fontsize=16)
        plt.xlabel('Engine Rating (Elo)', fontsize=12)
        plt.ylabel('Average Hit Score', fontsize=12)
        plt.legend(title='Engines & Trend')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Annotate each point with the engine name for easy identification.
        for i, row in analysis_df.iterrows():
            plt.text(row['rating'] + 5, row['average_hit_score'], row['engine_name'], fontsize=9)

        plt.savefig(OUTPUT_GRAPH_PATH, bbox_inches='tight')
        print(f"Successfully generated analysis graph: '{OUTPUT_GRAPH_PATH}'")
    except Exception as e:
        print(f"\nError creating graph: {e}")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    analyze_rating_log()
