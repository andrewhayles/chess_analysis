import chess
import chess.pgn
import chess.engine
import pandas as pd
import time
import sys
import os
import re
import io
from multiprocessing import Pool, Manager
from scipy.stats import linregress
from tqdm import tqdm
import subprocess

# ==============================================================================
# --- Tuning Configuration ---
# ==============================================================================
PARAMETER_SETS_TO_TEST = [
    {
        "name": "Top 1 Move Only",
        "num_moves": 1,
        "weights": [1.0]
    },
    {
        "name": "Top 3 Moves, Linear Weights",
        "num_moves": 3,
        "weights": [1.0, 0.5, 0.25]
    },
    {
        "name": "Top 5 Moves, Steep Decay",
        "num_moves": 5,
        "weights": [1.0, 0.4, 0.2, 0.1, 0.05]
    },
]

# ==============================================================================
# --- General Configuration ---
# ==============================================================================
ORACLE_ENGINE_NAME = "stockfish_full_1"
NUM_PARALLEL_GAMES = 1
START_MOVE = 10
MAX_GAMES_TO_PROCESS = 20
POSITIONS_PER_GAME = 5

PGN_FILE_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/TCEC games/TCEC.pgn"
ENGINES_CSV_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/real_engines.csv"
OUTPUT_SUMMARY_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/hitcount_tuning/r_squared_tuning_summary.csv"
DETAILED_LOG_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/hitcount_tuning/real_time_analysis_log.csv"

ORACLE_ANALYSIS_DEPTH = 22
ORACLE_ANALYSIS_TIMEOUT = 600
ENGINE_THREADS = 2

TEST_ANALYSIS_DEPTH = 12
TEST_ANALYSIS_TIMEOUT = 30

# ==============================================================================
# --- Core Logic ---
# ==============================================================================

def open_engine(path):
    """Opens a chess engine, suppressing stderr for specific engines like Leela."""
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    stderr_pipe = subprocess.DEVNULL if "leela" in path.lower() else None
    return chess.engine.SimpleEngine.popen_uci(path, stderr=stderr_pipe, startupinfo=startupinfo)

def listener_process(queue, file_path):
    """Listens for messages on the queue and writes them to the log file."""
    header_written = os.path.exists(file_path)
    with open(file_path, 'a', newline='') as f:
        while True:
            message = queue.get()
            if message == 'kill':
                break
            df = pd.DataFrame([message])
            df.to_csv(f, header=not header_written, index=False)
            f.flush()  # *** FIX: Force write to disk immediately ***
            header_written = True

def process_item(task_params):
    """
    Analyzes a single item (one position, one test engine) and puts the result on the queue.
    """
    board_fen, oracle_analysis, engine_info, game_id, move_number, run_name, weights, log_queue = task_params
    
    board = chess.Board(board_fen)
    test_engine = None
    try:
        test_engine_name = engine_info['engine_name']
        test_engine_path = engine_info['path']

        if not isinstance(test_engine_path, str) or not test_engine_path:
            return
        if not os.path.exists(test_engine_path):
            raise FileNotFoundError(f"Test engine '{test_engine_name}' not found at path: {test_engine_path}")
        
        test_engine = open_engine(test_engine_path)
        test_engine.configure({"Threads": ENGINE_THREADS})
        test_limit = chess.engine.Limit(depth=TEST_ANALYSIS_DEPTH, time=TEST_ANALYSIS_TIMEOUT)
        test_result = test_engine.play(board, test_limit)
        
        score_for_move = 0.0
        for i in range(len(oracle_analysis)):
            if test_result.move == oracle_analysis[i]['pv'][0]:
                score_for_move = weights[i]
                break
        
        log_queue.put({
            "run_name": run_name,
            "game_id": game_id,
            "move_number": move_number,
            "player_engine": test_engine_name,
            "score": score_for_move
        })

    except Exception as e:
        print(f"Error in worker for item {board_fen} / {engine_info['engine_name']}: {e}", file=sys.stderr)
    finally:
        if test_engine:
            test_engine.quit()

def main_tuner():
    print("--- Starting R-squared Optimization Tuning Script ---")
    
    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
        for col in ['engine_name', 'rating', 'path']:
            if col not in engines_df.columns: raise ValueError(f"Engines CSV must contain a '{col}' column.")
        
        oracle_row = engines_df[engines_df['engine_name'] == ORACLE_ENGINE_NAME]
        if oracle_row.empty: raise ValueError(f"Oracle engine '{ORACLE_ENGINE_NAME}' not found in '{ENGINES_CSV_PATH}'.")
        
        oracle_engine_path = oracle_row.iloc[0]['path']
        all_engines_info = engines_df[engines_df['engine_name'] != ORACLE_ENGINE_NAME].to_dict('records')
        
        print(f"Loaded {len(engines_df)} engines. Using '{ORACLE_ENGINE_NAME}' as the oracle.")

    except Exception as e:
        print(f"Error loading or parsing '{ENGINES_CSV_PATH}': {e}", file=sys.stderr); return

    print("Pre-processing games to generate positions...")
    positions_to_analyze = []
    with open(PGN_FILE_PATH, 'r', errors='ignore') as pgn:
        game_count = 0
        while True:
            if MAX_GAMES_TO_PROCESS and game_count >= MAX_GAMES_TO_PROCESS: break
            try:
                game = chess.pgn.read_game(pgn)
                if game is None: break
                game_count += 1
                board = game.board()
                positions_in_game = 0
                for move in game.mainline_moves():
                    if board.fullmove_number >= START_MOVE:
                        if POSITIONS_PER_GAME is not None and positions_in_game >= POSITIONS_PER_GAME: break
                        positions_to_analyze.append({"fen": board.fen(), "game_id": game.headers.get("Site", f"Game_{game_count}"), "move_number": board.fullmove_number})
                        positions_in_game += 1
                    board.push(move)
            except Exception: continue
    print(f"Generated {len(positions_to_analyze)} positions to analyze.")

    if not positions_to_analyze: print("No positions generated. Exiting."); return

    manager = Manager()
    log_queue = manager.Queue()
    listener = Pool(1).apply_async(listener_process, (log_queue, DETAILED_LOG_PATH))

    completed_items = set()
    if os.path.exists(DETAILED_LOG_PATH):
        print(f"Found existing log file. Checking for completed work...")
        log_df = pd.read_csv(DETAILED_LOG_PATH)
        for _, row in log_df.iterrows():
            # Create a unique identifier for each completed item
            completed_items.add((row['run_name'], row['game_id'], row['move_number'], row['player_engine']))
        print(f"Found {len(completed_items)} already completed analysis items.")

    start_time = time.time()
    oracle_engine = open_engine(oracle_engine_path)
    oracle_engine.configure({"Threads": ENGINE_THREADS})

    for param_set in PARAMETER_SETS_TO_TEST:
        run_name = param_set["name"]
        num_moves = param_set["num_moves"]
        weights = param_set["weights"]
        
        tasks = []
        print(f"\n--- Preparing tasks for run: '{run_name}' ---")
        for pos in tqdm(positions_to_analyze, desc="Pre-calculating Oracle moves"):
            oracle_limit = chess.engine.Limit(depth=ORACLE_ANALYSIS_DEPTH, time=ORACLE_ANALYSIS_TIMEOUT)
            oracle_analysis = oracle_engine.analyse(chess.Board(pos['fen']), oracle_limit, multipv=num_moves)
            
            for engine_info in all_engines_info:
                if (run_name, pos['game_id'], pos['move_number'], engine_info['engine_name']) not in completed_items:
                    tasks.append((pos['fen'], oracle_analysis, engine_info, pos['game_id'], pos['move_number'], run_name, weights, log_queue))
        
        if not tasks:
            print(f"All items for run '{run_name}' are already complete. Skipping.")
            continue
        
        print(f"Analyzing {len(tasks)} new items for '{run_name}'...")
        with Pool(processes=NUM_PARALLEL_GAMES) as pool:
            list(tqdm(pool.imap_unordered(process_item, tasks), total=len(tasks), desc=f"Run '{run_name}'"))

    oracle_engine.quit()
    log_queue.put('kill')
    listener.get()

    print("\n--- Generating Final Summary Report ---")
    all_summary_results = []
    if os.path.exists(DETAILED_LOG_PATH):
        full_log_df = pd.read_csv(DETAILED_LOG_PATH)
        for param_set in PARAMETER_SETS_TO_TEST:
            run_name = param_set['name']
            run_specific_log = full_log_df[full_log_df['run_name'] == run_name]
            
            if run_specific_log.empty: r_squared = 0.0
            else:
                engine_scores = {e['engine_name']: [] for e in all_engines_info}
                for _, row in run_specific_log.iterrows():
                    if row['player_engine'] in engine_scores:
                        engine_scores[row['player_engine']].append(row['score'])
                
                avg_scores = {name: sum(scores) / len(scores) if scores else 0 for name, scores in engine_scores.items()}
                results_df = pd.DataFrame(list(avg_scores.items()), columns=['engine_name', 'average_hit_score'])
                final_df = pd.merge(engines_df, results_df, on='engine_name')
                final_df.dropna(subset=['rating', 'average_hit_score'], inplace=True)
                
                if len(final_df) < 2: r_squared = 0.0
                else: r_squared = linregress(final_df['rating'], final_df['average_hit_score']).rvalue ** 2

            all_summary_results.append({"run_name": run_name, "r_squared": r_squared, "num_moves": param_set["num_moves"], "weights": str(param_set["weights"])})

    print("\n\n" + "=" * 60)
    print("--- Tuning Complete ---")
    print(f"Total time elapsed: {(time.time() - start_time) / 60:.2f} minutes")
    print("=" * 60)

    if not all_summary_results: print("No results were generated."); return

    summary_df = pd.DataFrame(all_summary_results)
    summary_df.sort_values(by="r_squared", ascending=False, inplace=True)
    
    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)
    print(f"\nFinal summary saved to '{OUTPUT_SUMMARY_PATH}'")

    print("\n--- Best Parameter Sets (Optimized for R-squared) ---")
    print(summary_df.to_string(index=False))
    print("\nScript finished.")

if __name__ == "__main__":
    main_tuner()
