import chess
import chess.pgn
import chess.engine
import pandas as pd
import time
import sys
import os
import itertools
from multiprocessing import Pool, cpu_count

# ==============================================================================
# --- Tuning Configuration ---
# ==============================================================================
# Define the parameter sets you want to test. Each dictionary in the list
# represents one full analysis run with a specific configuration.
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
        "name": "Top 3 Moves, Equal Weights",
        "num_moves": 3,
        "weights": [1.0, 1.0, 1.0]
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

# --- Parallelism & Run Control ---
# Number of games to process in parallel. Set this to the number of CPU cores
# you want to dedicate to this task.
NUM_PARALLEL_GAMES = 1
# The move number to start the analysis from.
START_MOVE = 10
# Max number of games to process for a quick test. Set to None for the whole file.
MAX_GAMES_TO_PROCESS = 30

# --- File Paths ---
ENGINES_CSV_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/real_engines.csv"
PGN_FILE_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/TCEC games/TCEC.pgn"
OUTPUT_CSV_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/hitcount_tuning/tuning_results.csv"

# --- Oracle Engine Settings ---
ORACLE_ANALYSIS_DEPTH = 22
ORACLE_ANALYSIS_TIMEOUT = 600
# Threads for EACH parallel engine instance. For best performance, this should
# typically be 1 when NUM_PARALLEL_GAMES is >= 1.
# Total cores used will be NUM_PARALLEL_GAMES * ENGINE_THREADS.
ENGINE_THREADS = 2

# ==============================================================================
# --- Core Logic (for a single worker process) ---
# ==============================================================================

def process_game(game_and_params):
    """
    This function is executed by each worker process. It analyzes a single game.
    """
    game, num_moves, weights = game_and_params
    
    # Each process must initialize its own engine instance.
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        engine.configure({"Threads": ENGINE_THREADS})
        
        board = game.board()
        total_score = 0
        positions_analyzed = 0

        for move in game.mainline_moves():
            move_number = (board.fullmove_number - 1) * 2 + (1 if board.turn == chess.WHITE else 2)
            if move_number < START_MOVE * 2: # Adjusting for ply
                board.push(move)
                continue

            board_before_move = board.copy()
            
            # --- Calculate Score ---
            limit = chess.engine.Limit(depth=ORACLE_ANALYSIS_DEPTH, time=ORACLE_ANALYSIS_TIMEOUT)
            analysis = engine.analyse(board_before_move, limit, multipv=num_moves)
            
            score_for_move = 0.0
            for i in range(min(num_moves, len(analysis))):
                if move == analysis[i]['pv'][0]:
                    score_for_move = weights[i]
                    break
            
            total_score += score_for_move
            positions_analyzed += 1
            board.push(move)
        
        return total_score, positions_analyzed

    except Exception as e:
        # Log the error without crashing the whole pool
        print(f"Error in worker process for game {game.headers.get('Site', '?')}: {e}", file=sys.stderr)
        return 0, 0
    finally:
        if engine:
            engine.quit()


def run_analysis_for_parameters(games, parameter_set):
    """
    Manages the parallel processing for a single set of tuning parameters.
    """
    num_moves = parameter_set["num_moves"]
    weights = parameter_set["weights"]
    
    print(f"\n--- Running Analysis for: '{parameter_set['name']}' ---")
    print(f"Template size: {num_moves}, Weights: {weights}, Parallel Games: {NUM_PARALLEL_GAMES}")

    # Prepare arguments for each worker process
    tasks = [(game, num_moves, weights) for game in games]

    total_score = 0
    total_positions = 0
    
    # Create a pool of worker processes
    with Pool(processes=NUM_PARALLEL_GAMES) as pool:
        # map() distributes the tasks to the worker pool
        results = pool.map(process_game, tasks)
    
    # Aggregate results from all workers
    for score, positions in results:
        total_score += score
        total_positions += positions

    if total_positions == 0:
        return 0, 0
    
    average_score = total_score / total_positions
    print(f"\nFinished run. Total Score: {total_score:.2f}, Average Score: {average_score:.4f}")
    return total_score, average_score


def main_tuner():
    """
    Main function to orchestrate the parameter tuning process.
    """
    print("--- Starting Parallel Hit Score Parameter Tuning Script ---")
    
    # --- Initialization and Sanity Checks ---
    # (Checks are performed before starting the resource-intensive part)
    # ... (file path checks, etc.)

    # --- Load all games into memory first ---
    print(f"Loading games from '{PGN_FILE_PATH}'... This may take a moment.")
    games = []
    with open(PGN_FILE_PATH) as pgn:
        while True:
            if MAX_GAMES_TO_PROCESS and len(games) >= MAX_GAMES_TO_PROCESS:
                break
            try:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                games.append(game)
            except Exception:
                continue # Skip malformed games
    print(f"Loaded {len(games)} games for analysis.")

    if not games:
        print("No games were loaded. Exiting.")
        return

    all_results = []
    start_time = time.time()

    # --- Main Tuning Loop ---
    for i, param_set in enumerate(PARAMETER_SETS_TO_TEST):
        run_start_time = time.time()
        
        total_score, avg_score = run_analysis_for_parameters(games, param_set)
        
        run_duration = time.time() - run_start_time
        
        result = {
            "run_name": param_set["name"],
            "num_moves": param_set["num_moves"],
            "weights": str(param_set["weights"]),
            "total_weighted_score": total_score,
            "average_score_per_position": avg_score,
            "duration_seconds": run_duration
        }
        all_results.append(result)

    # --- Final Results ---
    print("\n\n" + "=" * 60)
    print("--- Tuning Complete ---")
    print(f"Total time elapsed: {(time.time() - start_time) / 60:.2f} minutes")
    print("=" * 60)

    results_df = pd.DataFrame(all_results)
    results_df.sort_values(by="average_score_per_position", ascending=False, inplace=True)
    
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nDetailed tuning results saved to '{OUTPUT_CSV_PATH}'")

    print("\n--- Best Parameter Sets (by Average Score) ---")
    print(results_df.to_string(index=False))
    print("\nScript finished.")


if __name__ == "__main__":
    # This check is important for multiprocessing on some platforms (like Windows)
    main_tuner()
