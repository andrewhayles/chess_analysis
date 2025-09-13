import chess
import chess.engine
import pandas as pd
import os
import sys
import json
import random
import subprocess

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- File Paths (Should match your main script) ---
ENGINES_CSV_PATH = "real_engines.csv"
ORACLE_CACHE_PATH = "oracle_cache.json"

# --- Test Parameters ---
# The number of random positions to select from your cache for the test.
NUM_POSITIONS_TO_TEST = 10 
# A specific timeout for this test. Should be long enough for Maia to load.
ANALYSIS_TIMEOUT = 2.0 
# The weights to use for scoring, matching your main script.
CHOSEN_METHOD_WEIGHTS = [1.0, 0.5, 0.25] 

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================

def open_engine(path):
    """Opens a chess engine, handling potential startup issues."""
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    
    # We want to see errors for this test, so don't suppress stderr
    stderr_pipe = None 
    try:
        return chess.engine.SimpleEngine.popen_uci(path, stderr=stderr_pipe, startupinfo=startupinfo)
    except Exception as e:
        print(f"  -> ERROR opening engine at {path}: {e}", file=sys.stderr)
        return None

def get_move_score(move_to_check, oracle_moves, weights):
    """Calculates the score for a move based on the oracle's ranking."""
    score = 0.0
    if not oracle_moves: return 0.0
    # Ensure we don't go out of bounds if oracle has fewer moves than weights
    for i, oracle_move in enumerate(oracle_moves[:len(weights)]):
        if move_to_check == oracle_move:
            score = weights[i]
            break
    return score

# ==============================================================================
# --- Main Test Logic ---
# ==============================================================================

def run_maia_check():
    """Main function to run the quick check on Maia engines."""
    print("--- Starting Maia Engine Quick Check ---")

    # 1. Load Engines CSV and filter for Maia engines
    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
        maia_engines_df = engines_df[engines_df['engine_name'].str.contains("maia", case=False)].copy()
        if maia_engines_df.empty:
            print(f"Error: No engines with 'maia' in their name found in '{ENGINES_CSV_PATH}'.")
            return
        print(f"Found {len(maia_engines_df)} Maia engines to test.")
    except FileNotFoundError:
        print(f"Error: Engines CSV file not found at '{ENGINES_CSV_PATH}'. Exiting.", file=sys.stderr)
        return

    # 2. Load Oracle Cache to get test positions
    try:
        with open(ORACLE_CACHE_PATH, 'r') as f:
            oracle_cache = json.load(f)
        if len(oracle_cache) < NUM_POSITIONS_TO_TEST:
            print(f"Warning: Oracle cache has fewer than {NUM_POSITIONS_TO_TEST} positions. Testing on all {len(oracle_cache)} available positions.")
            test_fens = list(oracle_cache.keys())
        else:
            test_fens = random.sample(list(oracle_cache.keys()), NUM_POSITIONS_TO_TEST)
        print(f"Selected {len(test_fens)} random positions for the test.\n")
    except FileNotFoundError:
        print(f"Error: Oracle cache not found at '{ORACLE_CACHE_PATH}'. Cannot select positions.", file=sys.stderr)
        return
    except json.JSONDecodeError:
        print(f"Error: Oracle cache at '{ORACLE_CACHE_PATH}' is corrupted.", file=sys.stderr)
        return

    # 3. Iterate through each Maia engine and test it
    results = []
    for index, engine_row in maia_engines_df.iterrows():
        engine_name = engine_row['engine_name']
        engine_path = engine_row['path']
        print(f"--- Testing Engine: {engine_name} ---")

        engine = open_engine(engine_path)
        if not engine:
            print("  -> Skipping this engine due to startup failure.\n")
            continue

        total_score = 0.0
        positions_analyzed = 0
        
        try:
            # Set a standard analysis limit
            analysis_limit = chess.engine.Limit(time=ANALYSIS_TIMEOUT)
            
            for fen in test_fens:
                board = chess.Board(fen)
                oracle_moves = oracle_cache.get(fen, [])
                if not oracle_moves:
                    continue # Skip if FEN has no oracle moves cached

                try:
                    result = engine.play(board, analysis_limit)
                    move = result.move.uci()
                    score = get_move_score(move, oracle_moves, CHOSEN_METHOD_WEIGHTS)
                    total_score += score
                    positions_analyzed += 1
                    # print(f"  FEN: {fen[:20]}... | Move: {move} | Score: {score:.2f}") # Uncomment for verbose output
                except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
                    print(f"  -> Analysis failed for FEN {fen}. Error: {e}", file=sys.stderr)
            
            if positions_analyzed > 0:
                average_score = total_score / positions_analyzed
                results.append({'engine_name': engine_name, 'rating': engine_row['rating'], 'average_score': average_score})
                print(f"  -> Completed. Average Score: {average_score:.4f}\n")
            else:
                print("  -> No positions were successfully analyzed.\n")

        finally:
            if engine:
                engine.quit()

    # 4. Print Final Summary
    print("--- Final Results Summary ---")
    if not results:
        print("No engines were successfully tested.")
        return
        
    # Sort results by rating for clarity
    sorted_results = sorted(results, key=lambda x: x['rating'])
    for res in sorted_results:
        print(f"  {res['engine_name']} (Rating: {res['rating']}):\tAverage Score = {res['average_score']:.4f}")

    print("\nCheck that scores are non-zero and generally increase with the engine rating.")
    print("--- Script Finished ---")


if __name__ == "__main__":
    run_maia_check()
