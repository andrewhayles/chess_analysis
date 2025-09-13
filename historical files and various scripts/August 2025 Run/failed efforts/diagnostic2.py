import pandas as pd
import sys
import os
import json
import random

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- File Paths ---
GRANULAR_LOG_PATH = "granular_analysis_log.csv"
ENGINES_CSV_PATH = "real_engines.csv"
ORACLE_CACHE_PATH = "oracle_cache.json" # Needed for forensic analysis

# --- Names to Exclude from the Benchmark Set ---
PLAYER_NAME = "player"
ORACLE_ENGINE_NAME = "stockfish_full_1"
ENGINES_TO_EXCLUDE = []

# --- Scoring Weights (from your main script) ---
CHOSEN_METHOD = {
    "weights": [1.0, 0.5, 0.25]
}

# ==============================================================================
# --- Diagnostic Logic ---
# ==============================================================================

def get_move_score_forensic(move_to_check, oracle_moves, weights):
    """
    This is a copy of the scoring function from your main script.
    It returns the score for a move based on the oracle's ranking.
    """
    if not oracle_moves: return 0.0
    try:
        # Find the index of the move in the oracle's list
        index = oracle_moves.index(move_to_check)
        # If the index is within the bounds of our weights list, return the score
        if index < len(weights):
            return weights[index]
    except ValueError:
        # This happens if the move is not found in the oracle's list
        return 0.0
    return 0.0


def inspect_data():
    """
    Loads analysis data and performs a detailed forensic check on the scoring
    for a few sample positions to identify potential errors.
    """
    print("--- Forensic Data Inspector ---")

    # --- 1. Load Data ---
    try:
        log_df = pd.read_csv(GRANULAR_LOG_PATH)
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
        with open(ORACLE_CACHE_PATH, 'r') as f:
            oracle_cache = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e}. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(log_df)} log entries, {len(engines_df)} engine profiles, and {len(oracle_cache)} oracle positions.")

    # --- 2. Identify Benchmark Engines ---
    benchmark_names = engines_df[
        ~engines_df['engine_name'].isin([PLAYER_NAME, ORACLE_ENGINE_NAME] + ENGINES_TO_EXCLUDE)
    ]['engine_name'].unique()
    
    benchmark_log_df = log_df[log_df['engine_name'].isin(benchmark_names)]

    if benchmark_log_df.empty:
        print("\nCRITICAL ERROR: No benchmark engine data found in the log file.")
        return

    # --- 3. Perform Forensic Analysis on Samples ---
    print("\n--- Forensic Scoring Analysis (Random Samples) ---")
    
    # We need to find the actual move played by the engine for a given FEN.
    # The original log doesn't store this, so we'll simulate the scoring.
    # We need a way to get the engine's move. Since it's not in the log, we'll
    # just check the player's move for a few positions as a proxy.
    
    player_df = log_df[log_df['engine_name'] == 'player'].copy()
    if player_df.empty:
        print("\nCould not find player data to run forensic check. Please run the 'add_player_scores' script first.")
        return
        
    # Let's find the actual moves from your PGN file to test the logic
    try:
        from rating_estimator_new_dualcore import get_positions_from_pgn
        # This assumes the main script is in the same directory or accessible.
        # Ensure PLAYER_NAME_IN_PGN, PLAYER_PGN_PATH etc. are defined or passed.
        all_player_positions = get_positions_from_pgn(
            "chessgames_august2025.pgn", "Desjardins373", 5, 10
        )
        player_moves_map = {pos['fen']: pos['actual_move'] for pos in all_player_positions}
    except (ImportError, FileNotFoundError):
        print("\nWarning: Could not load actual moves from PGN. Forensic check will be less effective.")
        player_moves_map = {}


    sample_fens = random.sample(list(player_moves_map.keys()), min(5, len(player_moves_map)))

    for fen in sample_fens:
        actual_move = player_moves_map.get(fen)
        oracle_moves = oracle_cache.get(fen)
        
        if not actual_move or not oracle_moves:
            continue
            
        calculated_score = get_move_score_forensic(actual_move, oracle_moves, CHOSEN_METHOD['weights'])
        
        print(f"\nPosition (FEN): {fen}")
        print(f"  - Your Move:         '{actual_move}'")
        print(f"  - Oracle's Top Moves: {oracle_moves}")
        print(f"  - Calculated Score:   {calculated_score}")
        if actual_move in oracle_moves:
            print("  - DIAGNOSIS:         MATCH FOUND! The scoring logic appears to work correctly here.")
        else:
            print("  - DIAGNOSIS:         NO MATCH. The score of 0.0 is correct for this position.")

    print("\n--------------------------------------------------")
    print("Forensic check complete. Review the samples above.")


if __name__ == "__main__":
    inspect_data()
