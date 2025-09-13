import chess
import chess.pgn
import chess.engine
import pandas as pd
import time
import sys
import os
import re
import io
import json
from multiprocessing import Pool, Manager
from scipy.stats import linregress
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- Player and Game Configuration ---
PLAYER_NAME_IN_PGN = "Desjardins373" 
PLAYER_PGN_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/all rapid games since making goal from chess.com/goalgames.pgn" 

# --- Optimal Method Configuration ---
CHOSEN_METHOD = {
    "name": "Top 3 Moves, Linear Weights",
    "num_moves": 3,
    "weights": [1.0, 0.5, 0.25]
}

# --- Analysis Control ---
START_MOVE = 10
POSITIONS_PER_GAME = 5 
# *** NEW ***: Number of parallel worker processes to use for analysis.
NUM_WORKERS = 1

# --- File Paths ---
ENGINES_CSV_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/real_engines.csv"
ORACLE_CACHE_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/hitcount_tuning/oracle_cache.json"
OUTPUT_GRAPH_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/hitcount_tuning/player_rating_estimate.png"
ESTIMATION_LOG_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/hitcount_tuning/rating_estimation_log.csv"

# --- Engine Settings ---
ORACLE_ENGINE_NAME = "stockfish_full_1"
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

def get_positions_from_pgn(pgn_path, player_name, positions_per_game):
    """Extracts a list of FENs from a PGN file for a specific player."""
    positions = []
    print(f"Processing PGN: {os.path.basename(pgn_path)} for player: {player_name}")
    with open(pgn_path, 'r', errors='ignore') as pgn:
        game_count = 0
        while True:
            try:
                game = chess.pgn.read_game(pgn)
                if game is None: break
                
                white_player = game.headers.get("White", "")
                black_player = game.headers.get("Black", "")
                if player_name not in white_player and player_name not in black_player:
                    continue

                game_count += 1
                board = game.board()
                positions_in_game = 0
                is_player_white = player_name in white_player

                for move in game.mainline_moves():
                    if (is_player_white and board.turn == chess.WHITE) or \
                       (not is_player_white and board.turn == chess.BLACK):
                        
                        if board.fullmove_number >= START_MOVE:
                            if positions_per_game is not None and positions_in_game >= positions_per_game: break
                            positions.append({"fen": board.fen(), "actual_move": move.uci()})
                            positions_in_game += 1
                    board.push(move)
            except Exception: continue
    return positions

def listener_process(queue, file_path):
    """Listens for messages on the queue and writes them to the log file."""
    header_written = os.path.exists(file_path) and os.path.getsize(file_path) > 0
    with open(file_path, 'a', newline='') as f:
        while True:
            message = queue.get()
            if message == 'kill':
                break
            df = pd.DataFrame([message])
            df.to_csv(f, header=not header_written, index=False)
            f.flush()
            header_written = True

def analyze_benchmark_item(task_params):
    """Worker function to analyze a single position with a single test engine."""
    engine_info, position_fen, oracle_moves, num_moves, weights, log_queue = task_params
    engine = None
    try:
        engine_path = engine_info['path']
        if not isinstance(engine_path, str) or not os.path.exists(engine_path): return
        
        engine = open_engine(engine_path)
        engine.configure({"Threads": ENGINE_THREADS})
        
        board = chess.Board(position_fen)
        limit = chess.engine.Limit(depth=TEST_ANALYSIS_DEPTH, time=TEST_ANALYSIS_TIMEOUT)
        result = engine.play(board, limit)
        
        score = 0.0
        for i, oracle_move in enumerate(oracle_moves):
            if result.move.uci() == oracle_move:
                score = weights[i]
                break
        
        log_queue.put({
            "fen": position_fen,
            "engine_name": engine_info['engine_name'],
            "score": score
        })

    except Exception as e:
        print(f"Error in worker for {engine_info['engine_name']} on {position_fen}: {e}", file=sys.stderr)
    finally:
        # *** FIX: Handle unresponsive engines gracefully to prevent crashes. ***
        if engine:
            try:
                engine.quit()
            except (TimeoutError, chess.engine.EngineTerminatedError):
                # This can happen if an engine process is unresponsive or has already crashed.
                # We can ignore it and let the worker process finish.
                pass

def main():
    print("--- Starting Player Rating Estimation Script ---")
    
    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
        oracle_row = engines_df[engines_df['engine_name'] == ORACLE_ENGINE_NAME]
        if oracle_row.empty: raise ValueError(f"Oracle engine '{ORACLE_ENGINE_NAME}' not found.")
        oracle_engine_path = oracle_row.iloc[0]['path']
        benchmark_engines = engines_df[engines_df['engine_name'] != ORACLE_ENGINE_NAME].to_dict('records')
    except Exception as e:
        print(f"Error loading engine data: {e}"); return

    player_positions = get_positions_from_pgn(PLAYER_PGN_PATH, PLAYER_NAME_IN_PGN, POSITIONS_PER_GAME)
    if not player_positions:
        print(f"No positions found for player '{PLAYER_NAME_IN_PGN}' in '{PLAYER_PGN_PATH}'. Exiting."); return
    print(f"Found {len(player_positions)} positions for player '{PLAYER_NAME_IN_PGN}'.")

    oracle_cache = {}
    if os.path.exists(ORACLE_CACHE_PATH):
        with open(ORACLE_CACHE_PATH, 'r') as f:
            try: oracle_cache = json.load(f)
            except json.JSONDecodeError: pass
    
    positions_needing_oracle = [p for p in player_positions if p['fen'] not in oracle_cache]
    if positions_needing_oracle:
        print(f"Analyzing {len(positions_needing_oracle)} new positions with Oracle...")
        oracle_engine = open_engine(oracle_engine_path)
        oracle_engine.configure({"Threads": ENGINE_THREADS})
        for pos in tqdm(positions_needing_oracle, desc="Oracle Analysis"):
            board = chess.Board(pos['fen'])
            limit = chess.engine.Limit(depth=ORACLE_ANALYSIS_DEPTH, time=ORACLE_ANALYSIS_TIMEOUT)
            analysis = oracle_engine.analyse(board, limit, multipv=CHOSEN_METHOD['num_moves'])
            oracle_cache[pos['fen']] = [info['pv'][0].uci() for info in analysis]
        oracle_engine.quit()
        with open(ORACLE_CACHE_PATH, 'w') as f: json.dump(oracle_cache, f)
        print("Oracle cache updated.")

    # --- 1. Get Benchmark Scores using the player's positions ---
    manager = Manager()
    log_queue = manager.Queue()
    listener = Pool(1).apply_async(listener_process, (log_queue, ESTIMATION_LOG_PATH))

    completed_items = set()
    if os.path.exists(ESTIMATION_LOG_PATH):
        try:
            log_df = pd.read_csv(ESTIMATION_LOG_PATH)
            for _, row in log_df.iterrows():
                completed_items.add((row['fen'], row['engine_name']))
        except pd.errors.EmptyDataError:
            pass
    
    print(f"\nFound {len(completed_items)} previously completed benchmark items.")

    benchmark_tasks = []
    for engine in benchmark_engines:
        for pos in player_positions:
            if (pos['fen'], engine['engine_name']) not in completed_items:
                oracle_moves = oracle_cache.get(pos['fen'], [])[:CHOSEN_METHOD['num_moves']]
                benchmark_tasks.append((engine, pos['fen'], oracle_moves, CHOSEN_METHOD['num_moves'], CHOSEN_METHOD['weights'], log_queue))
    
    if benchmark_tasks:
        print(f"Calculating {len(benchmark_tasks)} new benchmark scores...")
        with Pool(processes=NUM_WORKERS) as pool:
            list(tqdm(pool.imap_unordered(analyze_benchmark_item, benchmark_tasks), total=len(benchmark_tasks), desc="Benchmarking Engines"))
    else:
        print("All benchmark scores are already calculated.")

    log_queue.put('kill')
    listener.get()

    # --- 2. Get Player Score ---
    print("\nCalculating player's hit score...")
    player_scores = []
    for pos in tqdm(player_positions, desc="Scoring Player's Moves"):
        oracle_moves = oracle_cache.get(pos['fen'], [])[:CHOSEN_METHOD['num_moves']]
        score = 0.0
        for i, oracle_move in enumerate(oracle_moves):
            if pos['actual_move'] == oracle_move:
                score = CHOSEN_METHOD['weights'][i]
                break
        player_scores.append(score)
    
    player_avg_score = sum(player_scores) / len(player_scores)
    print(f"Player '{PLAYER_NAME_IN_PGN}' average hit score: {player_avg_score:.4f}")

    # --- 3. Perform Regression and Estimate Rating ---
    if not os.path.exists(ESTIMATION_LOG_PATH) or os.path.getsize(ESTIMATION_LOG_PATH) == 0:
        print("Estimation log is empty. Cannot calculate rating."); return
        
    benchmark_log_df = pd.read_csv(ESTIMATION_LOG_PATH)
    avg_scores = benchmark_log_df.groupby('engine_name')['score'].mean().reset_index()
    avg_scores.rename(columns={'score': 'average_hit_score'}, inplace=True)

    final_df = pd.merge(engines_df, avg_scores, on='engine_name')
    
    if len(final_df) < 2:
        print("Not enough benchmark data to create a rating estimate."); return

    slope, intercept, r_value, _, _ = linregress(final_df['average_hit_score'], final_df['rating'])
    r_squared = r_value ** 2
    
    estimated_rating = (slope * player_avg_score) + intercept
    print(f"\nEstimated Rating for {PLAYER_NAME_IN_PGN}: {estimated_rating:.0f}")
    print(f"Benchmark R-squared: {r_squared:.4f}")

    # --- 4. Generate Graph ---
    plt.figure(figsize=(12, 8))
    sns.regplot(x='rating', y='average_hit_score', data=final_df, ci=None, line_kws={'color':'red', 'linestyle':'--'}, label='Engine Trend')
    plt.scatter(final_df['rating'], final_df['average_hit_score'], label='Benchmark Engines')
    plt.scatter(estimated_rating, player_avg_score, color='gold', s=200, edgecolor='black', zorder=5, label=f'You ({PLAYER_NAME_IN_PGN})')
    
    plt.title(f"Performance Analysis vs. Engine Rating\nMethod: {CHOSEN_METHOD['name']}", fontsize=16)
    plt.xlabel("Engine Rating", fontsize=12)
    plt.ylabel("Average Hit Score", fontsize=12)
    plt.legend()
    plt.grid(True)
    
    plt.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$\nEst. Rating: {estimated_rating:.0f}', 
             transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
             
    os.makedirs(os.path.dirname(OUTPUT_GRAPH_PATH), exist_ok=True)
    plt.savefig(OUTPUT_GRAPH_PATH)
    print(f"\nGraph saved to: {OUTPUT_GRAPH_PATH}")
    print("--- Script Finished ---")

if __name__ == "__main__":
    main()
