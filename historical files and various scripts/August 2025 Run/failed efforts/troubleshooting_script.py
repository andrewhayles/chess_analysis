import chess
import chess.pgn
import chess.engine
import pandas as pd
import sys
import os
import json
import random
import multiprocessing
from tqdm import tqdm
import subprocess
import concurrent.futures

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
# --- Configuration (Should match your original script) ---
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_

# --- Player and Game Configuration ---
PLAYER_NAME_IN_PGN = "Desjardins373"
PLAYER_PGN_PATH = "chessgames_august2025.pgn"

# --- Optimal Method Configuration ---
CHOSEN_METHOD = {
    "name": "Top 3 Moves, Linear Weights",
    "num_moves": 3,
    "weights": [1.0, 0.5, 0.25]
}

# --- Analysis Control ---
START_MOVE = 10
POSITIONS_PER_GAME = 5

# --- File Paths ---
ENGINES_CSV_PATH = "real_engines.csv"
GRANULAR_LOG_PATH = "granular_analysis_log.csv"
ORACLE_CACHE_PATH = "oracle_cache.json"

# --- Default Engine Settings ---
ENGINE_THREADS = 1 # CRITICAL: Set to 1 for parallel processing
DEFAULT_TEST_DEPTH = 9
DEFAULT_TEST_TIMEOUT = 0.05

# --- Engine-Specific Configuration Overrides ---
ENGINE_CONFIG_OVERRIDES = {
    "stockfish_elo_1950": {"depth": 4, "time": 0.05},
    "stockfish_elo_2007": {"depth": 4, "time": 0.05},
    "stockfish_elo_2050": {"depth": 4, "time": 0.05},
}

# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
# --- Worker Functions (Copied from original script) ---
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_

def open_engine(path):
    """Opens a chess engine, handling potential startup issues."""
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    # MODIFICATION: Always allow error messages to be shown for debugging.
    stderr_pipe = None 
    try:
        return chess.engine.SimpleEngine.popen_uci(path, stderr=stderr_pipe, startupinfo=startupinfo)
    except Exception as e:
        print(f"Error opening engine at {path}: {e}", file=sys.stderr)
        return None

def get_move_score(move_to_check, oracle_moves, weights):
    """Calculates the score for a move based on the oracle's ranking."""
    if not oracle_moves: return 0.0
    try:
        index = oracle_moves.index(move_to_check)
        if index < len(weights):
            return weights[index]
    except ValueError:
        return 0.0
    return 0.0

def analyze_position_worker(args):
    """
    Worker function to analyze a single position with a single benchmark engine.
    MODIFIED to handle unresponsive engines that cause TimeoutErrors on quit.
    """
    fen, engine_path, analysis_limit, oracle_moves, weights = args
    engine = None
    try:
        engine = open_engine(engine_path)
        if not engine:
            return {'fen': fen, 'score': 0.0}
        engine.configure({"Threads": ENGINE_THREADS})
        board = chess.Board(fen)
        result = engine.play(board, analysis_limit)
        score = get_move_score(result.move.uci(), oracle_moves, weights)
        return {'fen': fen, 'score': score}
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError, concurrent.futures.TimeoutError) as e:
        print(f"\nWarning: Engine at '{engine_path}' failed on FEN {fen}. Error: {e}", file=sys.stderr)
        return {'fen': fen, 'score': 0.0}
    finally:
        if engine:
            try:
                # First, try to quit gracefully.
                engine.quit()
            except (chess.engine.EngineError, concurrent.futures.TimeoutError):
                # If quitting hangs or fails, print a warning and then forcefully
                # terminate the process to prevent the entire script from crashing.
                print(f"\nWarning: Engine at '{engine_path}' was unresponsive during quit. Forcing shutdown.", file=sys.stderr)
                engine.close()


# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_
# --- Main Resumption Logic ---
# =_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_=_

def main():
    """Main function to resume analysis only for Maia engines."""
    print("--- Starting Maia Engine Analysis Resumption Script ---")

    # --- 1. Load Engine Data and Filter for Maia Engines ---
    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Engines CSV file not found at '{ENGINES_CSV_PATH}'. Exiting.", file=sys.stderr)
        return

    maia_engines_df = engines_df[engines_df['engine_name'].str.contains("maia", case=False, na=False)]
    if maia_engines_df.empty:
        print("No engines with 'maia' in their name found in 'real_engines.csv'. Exiting.")
        return
    
    engine_names_to_run = list(maia_engines_df['engine_name'])
    print(f"Found {len(engine_names_to_run)} Maia engines to process: {engine_names_to_run}")

    # --- 2. Load Caches and Game Positions ---
    if not os.path.exists(ORACLE_CACHE_PATH):
        print(f"Error: Oracle cache not found at '{ORACLE_CACHE_PATH}'. Cannot proceed."); return
    with open(ORACLE_CACHE_PATH, 'r') as f:
        oracle_cache = json.load(f)
    print(f"Loaded {len(oracle_cache)} positions from Oracle cache.")

    if not os.path.exists(PLAYER_PGN_PATH):
        print(f"Error: PGN file not found at '{PLAYER_PGN_PATH}'. Exiting.", file=sys.stderr)
        return
        
    # This is a bit of a hack to import from your main script.
    # Make sure 'rating_estimator_new_dualcore.py' is in the same folder.
    try:
        from rating_estimator_new_dualcore import get_positions_from_pgn 
        all_player_positions = get_positions_from_pgn(PLAYER_PGN_PATH, PLAYER_NAME_IN_PGN, POSITIONS_PER_GAME, START_MOVE)
        all_fens_in_scope = {p['fen'] for p in all_player_positions}
        print(f"Successfully loaded {len(all_fens_in_scope)} unique positions from PGN.")
    except ImportError:
        print("Could not import 'get_positions_from_pgn' from the main script.")
        print("Please ensure 'rating_estimator_new_dualcore.py' is in the same directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading positions from PGN: {e}")
        return


    # --- 3. Load Existing Log to Determine Work to Do ---
    completed_items = set()
    if os.path.exists(GRANULAR_LOG_PATH):
        try:
            log_df = pd.read_csv(GRANULAR_LOG_PATH)
            if not log_df.empty:
                completed_items = set(zip(log_df['fen'], log_df['engine_name']))
                print(f"Found {len(completed_items)} previously completed analyses in the log.")
        except (pd.errors.EmptyDataError, KeyError):
             print(f"'{GRANULAR_LOG_PATH}' is empty or malformed. Starting fresh.")
    
    # --- 4. Main Analysis Loop (Engine-by-Engine) ---
    for engine_name in engine_names_to_run:
        engine_details = maia_engines_df[maia_engines_df['engine_name'] == engine_name].iloc[0]
        
        # Identify tasks for THIS engine that are NOT in the completed log
        tasks_for_this_engine = []
        for fen in all_fens_in_scope:
            if (fen, engine_name) not in completed_items:
                config = ENGINE_CONFIG_OVERRIDES.get(engine_name, {})
                depth = config.get('depth', DEFAULT_TEST_DEPTH)
                timeout = config.get('time', DEFAULT_TEST_TIMEOUT)
                limit = chess.engine.Limit(depth=depth, time=timeout)
                tasks_for_this_engine.append(
                    (fen, engine_details['path'], limit, oracle_cache.get(fen, []), CHOSEN_METHOD['weights'])
                )
        
        if not tasks_for_this_engine:
            print(f"\nAll analyses for '{engine_name}' are already complete. Skipping.")
            continue

        print(f"\n--- Resuming analysis for '{engine_name}': {len(tasks_for_this_engine)} positions remaining ---")
        
        with multiprocessing.Pool(processes=2) as pool:
            results = list(tqdm(pool.imap_unordered(analyze_position_worker, tasks_for_this_engine), 
                                total=len(tasks_for_this_engine), 
                                desc=f"Engine: {engine_name}"))

        # Append new results to the granular log
        new_rows = [{'fen': res['fen'], 'engine_name': engine_name, 'score': res['score']} for res in results]
        
        if new_rows:
            new_rows_df = pd.DataFrame(new_rows)
            # Check if file is empty to write header, otherwise append without header
            is_new_file = not os.path.exists(GRANULAR_LOG_PATH) or os.path.getsize(GRANULAR_LOG_PATH) == 0
            new_rows_df.to_csv(GRANULAR_LOG_PATH, mode='a', header=is_new_file, index=False)
            print(f"Saved {len(new_rows_df)} new results for '{engine_name}' to the log.")

    print("\n--- Maia Engine Resumption Script Finished ---")
    print("You can now re-run your main analysis script to generate the final report and graph.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
