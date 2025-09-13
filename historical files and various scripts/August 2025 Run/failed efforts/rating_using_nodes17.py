import chess
import chess.engine
import chess.pgn
import json
import os
import random
from datetime import timedelta
import time

# --- Configuration ---
INPUT_PGN_FILE = 'chessgames_august2025.pgn' 
NEW_ORACLE_CACHE_FILE = 'oracle_cache_simple.json'
ORACLE_ENGINE_PATH = 'C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe'
ORACLE_SEARCH_NODES = 10_000_000
# MODIFIED: The number of new, simple positions to collect
NUM_POSITIONS_TO_SAMPLE = 500
MIN_MOVE_NUMBER = 8
MAX_MOVE_NUMBER = 20

# --- Main Script ---
def main():
    if not os.path.exists(INPUT_PGN_FILE):
        print(f"FATAL ERROR: Input PGN file not found at '{INPUT_PGN_FILE}'")
        return

    start_time = time.time()
    print("Starting oracle cache generation for 500 simple positions...")
    oracle_engine = chess.engine.SimpleEngine.popen_uci(ORACLE_ENGINE_PATH)
    oracle_engine.configure({"Threads": 16, "Hash": 1024})
    print("Oracle engine initialized.")

    positions_to_analyze = set()
    
    print(f"Scanning PGN file to gather at least {NUM_POSITIONS_TO_SAMPLE} unique positions...")
    with open(INPUT_PGN_FILE) as pgn:
        while len(positions_to_analyze) < NUM_POSITIONS_TO_SAMPLE:
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
    print(f"\n--- New simple oracle cache created successfully! ---")
    print(f"Saved {len(new_cache)} positions to '{NEW_ORACLE_CACHE_FILE}'")
    print(f"Total time: {timedelta(seconds=end_time - start_time)}")
    oracle_engine.quit()

if __name__ == "__main__":
    main()