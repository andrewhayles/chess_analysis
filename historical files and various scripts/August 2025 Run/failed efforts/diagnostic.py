import pandas as pd
import sys

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- File Paths ---
GRANULAR_LOG_PATH = "granular_analysis_log.csv"
ENGINES_CSV_PATH = "real_engines.csv"

# --- Names to Exclude from the Benchmark Set ---
PLAYER_NAME = "player"
ORACLE_ENGINE_NAME = "stockfish_full_1"
ENGINES_TO_EXCLUDE = []

# ==============================================================================
# --- Diagnostic Logic ---
# ==============================================================================

def inspect_data():
    """
    Loads the analysis data and prints a summary of the average scores
    for each benchmark engine to diagnose statistical issues.
    """
    print("--- Data Inspector ---")

    # --- 1. Load Data ---
    try:
        log_df = pd.read_csv(GRANULAR_LOG_PATH)
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(log_df)} log entries and {len(engines_df)} engine profiles.")

    # --- 2. Identify Benchmark Engines ---
    benchmark_names = engines_df[
        ~engines_df['engine_name'].isin([PLAYER_NAME, ORACLE_ENGINE_NAME] + ENGINES_TO_EXCLUDE)
    ]['engine_name'].unique()

    if len(benchmark_names) == 0:
        print("\nCRITICAL ERROR: No benchmark engines were found after filtering.")
        print("Please check your 'real_engines.csv' file and the exclusion lists in this script.")
        return

    print(f"\nFound {len(benchmark_names)} benchmark engines to inspect.")

    # --- 3. Calculate and Display Average Scores ---
    benchmark_log_df = log_df[log_df['engine_name'].isin(benchmark_names)]
    
    if benchmark_log_df.empty:
        print("\nCRITICAL ERROR: No entries for any benchmark engines were found in the log file.")
        print("This might indicate a naming mismatch between your log and your engines CSV.")
        return

    avg_scores = benchmark_log_df.groupby('engine_name')['score'].mean().reset_index()
    avg_scores.rename(columns={'score': 'average_score'}, inplace=True)

    print("\n--- Average Scores for Benchmark Engines ---")
    print(avg_scores.to_string(index=False))
    print("------------------------------------------")

    # --- 4. Final Diagnosis ---
    if avg_scores['average_score'].nunique() < 2:
        print("\nDIAGNOSIS: SUCCESS.")
        print("As suspected, all benchmark engines have the same average score.")
        print("This prevents a statistical correlation from being calculated.")
    else:
        print("\nDIAGNOSIS: UNEXPECTED RESULT.")
        print("The engines have different scores. There may be another issue.")

if __name__ == "__main__":
    inspect_data()
