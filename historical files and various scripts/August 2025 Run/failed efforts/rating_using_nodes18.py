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
EXISTING_LOG_FILE = 'granular_analysis_log_top3.csv'
COMBINED_LOG_FILE = 'granular_analysis_log_combined.csv'
DEFAULT_SEARCH_NODES = 1_000_000

# --- Stage 1: Oracle Cache Generation ---

def generate_simple_cache():
    """
    Generates a new oracle cache with simple positions from a PGN file.
    Skips if the cache file already exists.
    """
    if os.path.exists(NEW_ORACLE_CACHE_FILE):
        print(f"--- Stage 1: Skipped ---")
        print(f"'{NEW_ORACLE_CACHE_FILE}' already exists.\n")
        return

    print("--- Stage 1: Generate Simple Oracle Cache ---")
    if not os.path.exists(INPUT_PGN_FILE):
        print(f"FATAL ERROR: Input PGN file not found at '{INPUT_PGN_FILE}'")
        return

    start_time = time.time()
    print("Starting oracle cache generation for simple positions...")
    oracle_engine = None
    try:
        oracle_engine = chess.engine.SimpleEngine.popen_uci(ORACLE_ENGINE_PATH)
        oracle_engine.configure({"Threads": 16, "Hash": 1024})
        print("Oracle engine initialized.")

        positions_to_analyze = set()
        with open(INPUT_PGN_FILE) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                
                board = game.board()
                move_count = 0
                for move in game.mainline_moves():
                    move_count += 1
                    board.push(move)
                    if MIN_MOVE_NUMBER <= move_count <= MAX_MOVE_NUMBER:
                        positions_to_analyze.add(board.fen())
        
        final_positions = random.sample(list(positions_to_analyze), min(NUM_POSITIONS_TO_SAMPLE, len(positions_to_analyze)))
        print(f"Gathered {len(final_positions)} unique positions. Now analyzing...")

        new_cache = {}
        for i, fen in enumerate(final_positions):
            print(f"  Analyzing position {i+1}/{len(final_positions)}...")
            board = chess.Board(fen)
            try:
                analysis = oracle_engine.analyse(board, chess.engine.Limit(nodes=ORACLE_SEARCH_NODES), multipv=3)
                new_cache[fen] = [info['pv'][0].uci() for info in analysis]
            except Exception as e:
                print(f"    Error analyzing FEN {fen}: {e}")

        with open(NEW_ORACLE_CACHE_FILE, 'w') as f:
            json.dump(new_cache, f, indent=2)

        end_time = time.time()
        print(f"--- Stage 1 Complete ---")
        print(f"Saved {len(final_positions)} positions to '{NEW_ORACLE_CACHE_FILE}'")
        print(f"Total time for Stage 1: {timedelta(seconds=end_time - start_time)}\n")

    finally:
        if oracle_engine:
            oracle_engine.quit()


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

def run_engine_analysis():
    """
    Runs all configured engines against all positions in the oracle caches.
    """
    print("--- Stage 2: Granular Engine Analysis ---")
    
    # 1. Load all data and configurations
    engine_configs = load_engine_configs(ENGINES_CSV_FILE)
    if not engine_configs: return
        
    oracle_positions = load_oracle_positions(ORACLE_CACHE_FILES)
    existing_df, completed_analysis = get_completed_analysis(EXISTING_LOG_FILE)
    
    new_results = []
    total_new_analyses = 0

    # 2. Iterate through each engine
    for config in engine_configs:
        engine_name, engine_path, uci_options = config['name'], config['path'], config['uci_options']
        
        # --- FIX IS HERE ---
        # Pop 'Nodes' from the uci_options. It's not a configurable option,
        # but a parameter for the search limit.
        search_nodes = uci_options.pop('Nodes', DEFAULT_SEARCH_NODES)
        
        if not os.path.exists(engine_path):
            print(f"\n--- WARNING: Engine not found for '{engine_name}'. Skipping. ---")
            continue

        print(f"\n--- Analyzing with engine: {engine_name} (Nodes: {search_nodes}) ---")
        engine = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            # Only configure the engine if there are any actual options left
            if uci_options:
                engine.configure(uci_options)
            
            newly_analyzed_count = 0
            # 3. Iterate through each position
            for fen, oracle_moves in oracle_positions.items():
                if (engine_name, fen) in completed_analysis:
                    continue

                board = chess.Board(fen)
                try:
                    # Use the search_nodes value here in the limit
                    limit = chess.engine.Limit(nodes=int(search_nodes))
                    info = engine.analyse(board, limit, multipv=1)
                    engine_move = info[0]['pv'][0].uci()

                    move_rank = oracle_moves.index(engine_move) if engine_move in oracle_moves else -1
                    score = 1 if move_rank != -1 else 0
                    
                    new_results.append({'engine_name': engine_name, 'fen': fen, 'score': score, 'move_rank': move_rank})
                    newly_analyzed_count += 1
                    total_new_analyses += 1
                    
                    if newly_analyzed_count > 0 and newly_analyzed_count % 50 == 0:
                        print(f"  Analyzed {newly_analyzed_count} new positions for {engine_name}...")

                except Exception as e:
                    print(f"    ERROR analyzing FEN {fen}: {e}")

            print(f"Finished analysis for {engine_name}. Analyzed {newly_analyzed_count} new positions.")

        finally:
            if engine:
                engine.quit()

    # 4. Combine and save final results
    if new_results:
        print(f"\nAdding {len(new_results)} new analysis results to the log.")
        new_df = pd.DataFrame(new_results)[['engine_name', 'fen', 'score', 'move_rank']]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['engine_name', 'fen'], keep='last')
        combined_df.to_csv(COMBINED_LOG_FILE, index=False)
        print(f"Successfully saved combined analysis to '{COMBINED_LOG_FILE}'")
    else:
        print("\nNo new analysis was performed. The log file is already up to date.")
    
    print(f"Total new positions analyzed across all engines: {total_new_analyses}")
    print(f"--- Stage 2 Complete ---")

# --- Main Script ---

if __name__ == "__main__":
    script_start_time = time.time()
    
    # Run Stage 1
    generate_simple_cache()
    
    # Run Stage 2
    run_engine_analysis()
    
    script_end_time = time.time()
    print(f"\n--- Full Process Finished ---")
    print(f"Total script time: {timedelta(seconds=script_end_time - script_start_time)}")