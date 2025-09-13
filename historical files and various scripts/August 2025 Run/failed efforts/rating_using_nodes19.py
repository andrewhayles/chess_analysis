import chess
import chess.engine
import chess.pgn
import json
import os
import random
import pandas as pd
import time
from datetime import timedelta

# --- Configuration ---

# Stage 1: Oracle Cache Generation
INPUT_PGN_FILE = 'chessgames_august2025.pgn'
NEW_ORACLE_CACHE_FILE = 'oracle_cache_simple.json'
ORACLE_ENGINE_PATH = 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe'
ORACLE_SEARCH_NODES = 10_000_000
NUM_POSITIONS_TO_SAMPLE = 500
MIN_MOVE_NUMBER = 8
MAX_MOVE_NUMBER = 20

# Stage 2: Granular Engine Analysis
ENGINES_CSV_FILE = 'real_engines.csv'
ORACLE_CACHE_FILES = ['oracle_cache_simple.json', 'oracle_cache_top3.json']
# MODIFIED: The script will now read from and append directly to this file.
GRANULAR_LOG_FILE = 'granular_analysis_log_top3.csv'
DEFAULT_SEARCH_NODES = 1_000_000
SAVE_BATCH_SIZE = 50 # How often to save progress (after this many new analyses)

# --- Stage 1: Oracle Cache Generation ---
# (This section remains unchanged)
def generate_simple_cache():
    if os.path.exists(NEW_ORACLE_CACHE_FILE):
        print(f"--- Stage 1: Skipped ---")
        print(f"'{NEW_ORACLE_CACHE_FILE}' already exists.\n")
        return

    print("--- Stage 1: Generate Simple Oracle Cache ---")
    # ... (code for this function is identical to the previous version)

# --- Stage 2: Granular Engine Analysis ---

def load_engine_configs(csv_file):
    """Loads engine configurations from the CSV file."""
    if not os.path.exists(csv_file):
        print(f"FATAL ERROR: Engine configuration file not found: {csv_file}")
        return []
    
    print(f"Loading engine configurations from '{csv_file}'...")
    df = pd.read_csv(csv_file)
    configs = []
    for index, row in df.iterrows():
        try:
            uci_options_str = row['uci_options'].replace('""', '"')
            uci_options = json.loads(uci_options_str)
        except (TypeError, json.JSONDecodeError):
            uci_options = {}
        configs.append({'name': row['engine_name'], 'path': row['path'], 'uci_options': uci_options})
    print(f"Loaded {len(configs)} engine configurations.")
    return configs

def load_oracle_positions(cache_files):
    """Loads all unique positions from all specified oracle cache files."""
    all_positions = {}
    print("Loading oracle cache files...")
    for file_path in cache_files:
        if not os.path.exists(file_path):
            print(f"  WARNING: Cache file not found: {file_path}")
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)
            for fen, value in data.items():
                if fen not in all_positions:
                    if isinstance(value, list):
                        all_positions[fen] = value
                    elif isinstance(value, dict) and 'oracle_moves' in value:
                        all_positions[fen] = value['oracle_moves']
    print(f"Loaded {len(all_positions)} unique positions.")
    return all_positions

def get_completed_analysis(log_file):
    """Reads the existing log to find which positions have already been analyzed."""
    if not os.path.exists(log_file):
        print("No existing log file found. Starting fresh.")
        return pd.DataFrame(), set()

    print(f"Loading existing analysis from '{log_file}'...")
    df = pd.read_csv(log_file)
    completed_set = set(tuple(row) for row in df[['engine_name', 'fen']].to_numpy())
    print(f"Found {len(completed_set)} unique engine-position pairs already analyzed.")
    return df, completed_set

# --- NEW: Function to save progress ---
def append_to_log(new_results, log_file):
    """Appends a list of new results to the CSV log file."""
    if not new_results:
        return
    
    # Check if the file exists to determine if we need to write a header
    header = not os.path.exists(log_file)
    
    new_df = pd.DataFrame(new_results)
    new_df.to_csv(log_file, mode='a', header=header, index=False)
    print(f"  .. Saved progress for {len(new_results)} new positions.")

def run_engine_analysis():
    """
    Runs all configured engines against all positions in the oracle caches.
    """
    print("--- Stage 2: Granular Engine Analysis ---")
    
    engine_configs = load_engine_configs(ENGINES_CSV_FILE)
    if not engine_configs: return
        
    oracle_positions = load_oracle_positions(ORACLE_CACHE_FILES)
    # MODIFIED: No longer need to hold the existing dataframe in memory
    _, completed_analysis = get_completed_analysis(GRANULAR_LOG_FILE)
    
    results_since_last_save = []
    total_new_analyses = 0

    for config in engine_configs:
        engine_name, engine_path, uci_options = config['name'], config['path'], config['uci_options']
        search_nodes = uci_options.pop('Nodes', DEFAULT_SEARCH_NODES)
        
        if not os.path.exists(engine_path):
            print(f"\n--- WARNING: Engine not found for '{engine_name}'. Skipping. ---")
            continue

        print(f"\n--- Analyzing with engine: {engine_name} (Nodes: {search_nodes}) ---")
        engine = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            if uci_options:
                engine.configure(uci_options)
            
            newly_analyzed_count = 0
            for fen, oracle_moves in oracle_positions.items():
                if (engine_name, fen) in completed_analysis:
                    continue

                board = chess.Board(fen)
                try:
                    limit = chess.engine.Limit(nodes=int(search_nodes))
                    info = engine.analyse(board, limit, multipv=1)
                    engine_move = info[0]['pv'][0].uci()

                    move_rank = oracle_moves.index(engine_move) if engine_move in oracle_moves else -1
                    score = 1 if move_rank != -1 else 0
                    
                    # MODIFIED: Append to a temporary list
                    results_since_last_save.append({'engine_name': engine_name, 'fen': fen, 'score': score, 'move_rank': move_rank})
                    
                    newly_analyzed_count += 1
                    total_new_analyses += 1
                    
                    # MODIFIED: Save progress periodically
                    if len(results_since_last_save) >= SAVE_BATCH_SIZE:
                        append_to_log(results_since_last_save, GRANULAR_LOG_FILE)
                        results_since_last_save.clear()

                except Exception as e:
                    print(f"    ERROR analyzing FEN {fen}: {e}")

            print(f"Finished analysis for {engine_name}. Analyzed {newly_analyzed_count} new positions.")

        finally:
            if engine:
                engine.quit()

    # --- MODIFIED: Final save for any remaining results ---
    if results_since_last_save:
        print("\nPerforming final save...")
        append_to_log(results_since_last_save, GRANULAR_LOG_FILE)
        results_since_last_save.clear()
    
    print(f"\nTotal new positions analyzed across all engines: {total_new_analyses}")
    print(f"--- Stage 2 Complete ---")

# --- Main Script ---

if __name__ == "__main__":
    script_start_time = time.time()
    
    # generate_simple_cache() # You can comment this out if you don't need to run it
    run_engine_analysis()
    
    script_end_time = time.time()
    print(f"\n--- Full Process Finished ---")
    print(f"Total script time: {timedelta(seconds=script_end_time - script_start_time)}")