import chess
import chess.engine
import pandas as pd
import json
import time
import os
from datetime import timedelta

# --- Configuration ---
# NOTE: The PGN file is no longer used. Analysis is based on the oracle cache.
OLD_GRANULAR_LOG_FILE = 'granular_analysis_log_top3.csv'
NEW_GRANULAR_LOG_FILE = 'granular_analysis_log_updated.csv'
ORACLE_CACHE_FILE = 'oracle_cache_top3.json'

# MODIFIED: This engine list is now correct and matches the revised plan.
engine_configs = {
    'stockfish_nodes_5': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 5},
    'stockfish_nodes_10': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 10},
    'stockfish_nodes_15': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 15},
    'stockfish_nodes_20': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 20},
    'stockfish_nodes_30': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 30},
    'stockfish_nodes_40': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 40},
    'stockfish_nodes_50': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 50},
    'stockfish_nodes_200': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 200},
    'stockfish_nodes_500': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 500},
    'stockfish_nodes_2k': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 2000},
    'stockfish_nodes_5k': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 5000},
    'stockfish_nodes_10k': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 256, 'Threads': 8},'search_param': 'nodes','search_value': 10000},
    'stockfish_nodes_20k': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 256, 'Threads': 8},'search_param': 'nodes','search_value': 20000},
    'stockfish_nodes_50k': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 512, 'Threads': 16},'search_param': 'nodes','search_value': 50000},
    'stockfish_nodes_100k': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 512, 'Threads': 16},'search_param': 'nodes','search_value': 100000},
    'stockfish_nodes_250k': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 1024, 'Threads': 16},'search_param': 'nodes','search_value': 250000},
    'stockfish_nodes_1M': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 1024, 'Threads': 16},'search_param': 'nodes','search_value': 1000000},
    'dragon': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/dragon/dragon_05e2a7/Windows/dragon-64bit.exe','protocol': 'uci','options': {'Hash': 1024, 'Threads': 16},'search_param': 'nodes','search_value': 500000},
    'stockfish_full_1': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 1024, 'Threads': 16},'search_param': 'nodes','search_value': 5000000}
}
# --- Utility Functions ---
def load_oracle_cache():
    if not os.path.exists(ORACLE_CACHE_FILE):
        print(f"FATAL ERROR: Oracle cache file not found at '{ORACLE_CACHE_FILE}'")
        return None
    with open(ORACLE_CACHE_FILE, 'r') as f:
        cache = json.load(f)
        print(f"Loaded {len(cache)} positions from oracle cache to use as the test suite.")
        return cache

def get_analyzed_positions():
    analyzed = set()
    for file_path in [OLD_GRANULAR_LOG_FILE, NEW_GRANULAR_LOG_FILE]:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    analyzed.update(zip(df['fen'], df['engine_name']))
            except pd.errors.EmptyDataError:
                continue
    print(f"Found {len(analyzed)} previously completed analyses in log files.")
    return analyzed

def analyze_move(board, engine_name, player_move, oracle_top_moves):
    fen = board.fen()
    move_rank = oracle_top_moves.index(player_move.uci()) if player_move.uci() in oracle_top_moves else -1
    return {"fen": fen,"engine_name": engine_name,"player_move_uci": player_move.uci(),"move_rank": move_rank,"oracle_top_3_uci": oracle_top_moves}

def save_progress(new_results_list):
    if not new_results_list: return
    print(f"Saving {len(new_results_list)} new results to log...")
    new_df = pd.DataFrame(new_results_list)
    header = not os.path.exists(NEW_GRANULAR_LOG_FILE) or os.path.getsize(NEW_GRANULAR_LOG_FILE) == 0
    new_df.to_csv(NEW_GRANULAR_LOG_FILE, mode='a', header=header, index=False)
    print("...Save complete.")

# --- Main Execution ---
def main():
    oracle_cache = load_oracle_cache()
    if oracle_cache is None:
        return

    analyzed_positions = get_analyzed_positions()
    start_time = time.time()
    new_results = []
    
    total_jobs = len(oracle_cache) * len(engine_configs)
    completed_jobs = len(analyzed_positions)
    print(f"Total analysis jobs to complete: {total_jobs}. Already completed: {completed_jobs}.")

    try:
        for i, (fen, oracle_top_moves) in enumerate(oracle_cache.items()):
            board = chess.Board(fen)
            
            # Reduce verbose logging unless necessary
            if (i+1) % 10 == 0 or i == 0:
                print(f"\nProcessing position {i+1}/{len(oracle_cache)}: {fen}")

            for name, config in engine_configs.items():
                if (fen, name) in analyzed_positions:
                    continue

                test_engine = None
                try:
                    # Log only when starting a new engine on a position batch
                    if (i+1) % 10 == 0 or i == 0:
                         print(f"  -> Analyzing with {name}...")
                    test_engine = chess.engine.SimpleEngine.popen_uci(config['path'], timeout=30)
                    test_engine.configure(config['options'])
                    
                    limit = chess.engine.Limit(**{config['search_param']: config['search_value']})
                    result = test_engine.play(board, limit)
                    engine_move = result.move

                    log_entry = analyze_move(board, name, engine_move, oracle_top_moves)
                    new_results.append(log_entry)
                    analyzed_positions.add((fen, name))
                
                except Exception as e:
                    print(f"  !! Error analyzing with {name} on FEN {fen}: {e}")
                
                finally:
                    if test_engine:
                        test_engine.quit()

            if len(new_results) >= 50:
                save_progress(new_results)
                new_results = []

    finally:
        print("\nPerforming final save before exit...")
        save_progress(new_results)
        
        end_time = time.time()
        print("\n--- Analysis Complete ---")
        print(f"Total time: {timedelta(seconds=end_time - start_time)}")
        print(f"Results saved to '{NEW_GRANULAR_LOG_FILE}'")

if __name__ == "__main__":
    main()