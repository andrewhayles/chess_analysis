import chess
import chess.engine
import chess.pgn
import pandas as pd
import json
import time
import os
from datetime import timedelta

# --- Configuration ---
PLAYER_NAME = 'Desjardins373'
PGN_FILE_PATH = 'chessgames_august2025.pgn'
OLD_GRANULAR_LOG_FILE = 'granular_analysis_log_top3.csv'
NEW_GRANULAR_LOG_FILE = 'granular_analysis_log_updated.csv'
ORACLE_ENGINE_PATH = 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe'
ORACLE_SEARCH_NODES = 10_000_000
ORACLE_CACHE_FILE = 'oracle_cache_top3.json'

# Engine configurations remain the same
engine_configs = {
    'stockfish_elo_1320': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'UCI_LimitStrength': True, 'UCI_Elo': 1320, 'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 5000},
    'stockfish_elo_1400': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'UCI_LimitStrength': True, 'UCI_Elo': 1400, 'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 5000},
    'stockfish_elo_1500': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'UCI_LimitStrength': True, 'UCI_Elo': 1500, 'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 5000},
    'stockfish_elo_1600': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'UCI_LimitStrength': True, 'UCI_Elo': 1600, 'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 5000},
    'stockfish_elo_1700': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'UCI_LimitStrength': True, 'UCI_Elo': 1700, 'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 5000},
    'stockfish_elo_1800': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'UCI_LimitStrength': True, 'UCI_Elo': 1800, 'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 5000},
    'stockfish_elo_1900': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'UCI_LimitStrength': True, 'UCI_Elo': 1900, 'Hash': 128, 'Threads': 8},'search_param': 'nodes','search_value': 5000},
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
    'dragon': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/dragon/dragon_05e2a7/Windows/dragon-64bit.exe','protocol': 'uci','options': {'Hash': 2048, 'Threads': 16},'search_param': 'nodes','search_value': 500000},
    'stockfish_full_1': {'path': 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe','protocol': 'uci','options': {'Hash': 2048, 'Threads': 16},'search_param': 'nodes','search_value': 5000000}
}
# --- Utility Functions ---

def load_oracle_cache():
    if os.path.exists(ORACLE_CACHE_FILE):
        with open(ORACLE_CACHE_FILE, 'r') as f:
            cache = json.load(f)
            # NEW: Diagnostic print
            print(f"Loaded {len(cache)} positions from oracle cache.")
            return cache
    print("Oracle cache not found. A new one will be created.")
    return {}

def save_oracle_cache(cache):
    with open(ORACLE_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

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
    return analyzed

def get_oracle_analysis(board, oracle_engine, cache):
    fen = board.fen()
    # NEW: Diagnostic print to show what key is being looked up
    print(f"  Oracle lookup for FEN: '{fen}'")
    if fen in cache:
        return cache[fen]

    print(f"  --> FEN not in cache. Analyzing new position.")
    analysis = oracle_engine.analyse(board, chess.engine.Limit(nodes=ORACLE_SEARCH_NODES), multipv=3)
    top_moves = [info['pv'][0].uci() for info in analysis]
    cache[fen] = top_moves
    return top_moves

def analyze_move(board, engine_name, player_move, oracle_top_moves):
    fen = board.fen()
    move_rank = oracle_top_moves.index(player_move.uci()) if player_move.uci() in oracle_top_moves else -1
    return {"fen": fen,"engine_name": engine_name,"player_move_uci": player_move.uci(),"move_rank": move_rank,"oracle_top_3_uci": oracle_top_moves}

def save_progress(new_results_list, cache):
    if not new_results_list:
        return
    print(f"Saving {len(new_results_list)} new results to log...")
    new_df = pd.DataFrame(new_results_list)
    header = not os.path.exists(NEW_GRANULAR_LOG_FILE) or os.path.getsize(NEW_GRANULAR_LOG_FILE) == 0
    new_df.to_csv(NEW_GRANULAR_LOG_FILE, mode='a', header=header, index=False)
    save_oracle_cache(cache)
    print("...Save complete.")

# --- Main Execution ---

def main():
    if not os.path.exists(PGN_FILE_PATH):
        print(f"FATAL ERROR: PGN file not found at '{PGN_FILE_PATH}'")
        print("Please update the PGN_FILE_PATH variable in the script.")
        return

    print("Starting analysis...")
    oracle_cache = load_oracle_cache()
    analyzed_positions = get_analyzed_positions()
    
    try:
        oracle_engine = chess.engine.SimpleEngine.popen_uci(ORACLE_ENGINE_PATH)
        oracle_engine.configure({"Threads": 16})
        print("Oracle engine initialized.")
    except Exception as e:
        print(f"Failed to initialize oracle engine: {e}")
        return

    engines = {}
    for name, config in engine_configs.items():
        try:
            engine = chess.engine.SimpleEngine.popen_uci(config['path'])
            engine.configure(config['options'])
            engines[name] = engine
            print(f"Initialized engine: {name}")
        except Exception as e:
            # NEW: More detailed error message
            print(f"Failed to initialize engine {name}:")
            print(f"  - Path checked: {config['path']}")
            print(f"  - Exception Type: {type(e).__name__}")
            print(f"  - Error details: {e}")

    if not engines:
        print("No test engines could be initialized. Exiting.")
        oracle_engine.quit()
        return

    start_time = time.time()
    positions_processed_total = 0
    new_results = []

    try:
        with open(PGN_FILE_PATH) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None: break

                player_color = chess.WHITE if game.headers.get('White', '').lower() == PLAYER_NAME.lower() else (chess.BLACK if game.headers.get('Black', '').lower() == PLAYER_NAME.lower() else None)
                if player_color is None: continue

                board = game.board()
                for move in game.mainline_moves():
                    if board.turn == player_color:
                        positions_processed_total += 1
                        fen = board.fen()
                        
                        oracle_top_moves = get_oracle_analysis(board, oracle_engine, oracle_cache)

                        for name, engine in engines.items():
                            if (fen, name) in analyzed_positions: continue

                            config = engine_configs[name]
                            limit = chess.engine.Limit(**{config['search_param']: config['search_value']})
                            result = engine.play(board, limit)
                            engine_move = result.move

                            log_entry = analyze_move(board, name, engine_move, oracle_top_moves)
                            new_results.append(log_entry)
                            analyzed_positions.add((fen, name))

                            if len(new_results) >= 50:
                                save_progress(new_results, oracle_cache)
                                new_results = []

                    board.push(move)
    finally:
        print("\nPerforming final save before exit...")
        save_progress(new_results, oracle_cache)
        
        oracle_engine.quit()
        for engine in engines.values():
            engine.quit()
        
        end_time = time.time()
        print("\n--- Analysis Complete ---")
        print(f"Total positions scanned in PGN: {positions_processed_total}")
        print(f"Total time: {timedelta(seconds=end_time - start_time)}")
        print(f"Oracle cache contains {len(oracle_cache)} positions.")
        print(f"Results saved to '{NEW_GRANULAR_LOG_FILE}'")

if __name__ == "__main__":
    main()