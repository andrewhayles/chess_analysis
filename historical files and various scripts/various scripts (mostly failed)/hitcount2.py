import chess
import chess.pgn
import chess.engine
import pandas as pd
import time
import sys
import os
import re
from multiprocessing import Pool
from scipy.stats import linregress

# ==============================================================================
# --- Tuning Configuration ---
# ==============================================================================
# Define the parameter sets to test. The script will find which set produces
# the highest R-squared value between engine rating and hit score.
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
NUM_PARALLEL_GAMES = 1
START_MOVE = 10
MAX_GAMES_TO_PROCESS = 20

# --- File Paths ---
ENGINE_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/engines/leela_chess"
PGN_FILE_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/TCEC games/TCEC.pgn"
ENGINES_CSV_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/real_engines.csv" # CSV with 'engine_name' and 'rating' columns
OUTPUT_CSV_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/hitcount_tuning/r_squared_tuning_results.csv"

# --- Oracle Engine Settings ---
ORACLE_ANALYSIS_DEPTH = 22
ORACLE_ANALYSIS_TIMEOUT = 600
ENGINE_THREADS = 2 # Keep at 1 for parallel game processing

# ==============================================================================
# --- Core Logic (for a single worker process) ---
# ==============================================================================

def process_game(game_and_params):
    """
    Analyzes a single game and returns a list of (engine_name, score) tuples.
    This function is executed by each worker process.
    """
    game, num_moves, weights, target_engines = game_and_params
    
    engine = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        engine.configure({"Threads": ENGINE_THREADS})
        
        board = game.board()
        results = []
        
        white_player = game.headers.get("White", "Unknown")
        black_player = game.headers.get("Black", "Unknown")

        for move in game.mainline_moves():
            player_to_move = white_player if board.turn() == chess.WHITE else black_player
            
            move_number = (board.fullmove_number - 1) * 2 + (1 if board.turn == chess.WHITE else 2)
            if move_number < START_MOVE * 2:
                board.push(move)
                continue

            # Only analyze if the player is one of the engines we are testing
            if player_to_move in target_engines:
                board_before_move = board.copy()
                limit = chess.engine.Limit(depth=ORACLE_ANALYSIS_DEPTH, time=ORACLE_ANALYSIS_TIMEOUT)
                analysis = engine.analyse(board_before_move, limit, multipv=num_moves)
                
                score_for_move = 0.0
                for i in range(min(num_moves, len(analysis))):
                    if move == analysis[i]['pv'][0]:
                        score_for_move = weights[i]
                        break
                results.append((player_to_move, score_for_move))

            board.push(move)
        
        return results

    except Exception as e:
        print(f"Error in worker for game {game.headers.get('Site', '?')}: {e}", file=sys.stderr)
        return []
    finally:
        if engine:
            engine.quit()

def run_analysis_for_parameters(games, engines_df, parameter_set):
    """
    Manages parallel processing and calculates the final R-squared value for a parameter set.
    """
    num_moves = parameter_set["num_moves"]
    weights = parameter_set["weights"]
    target_engines = set(engines_df['engine_name'])
    
    print(f"\n--- Running Analysis for: '{parameter_set['name']}' ---")
    print(f"Template size: {num_moves}, Weights: {weights}")

    tasks = [(game, num_moves, weights, target_engines) for game in games]
    
    # This dictionary will hold all scores for each engine: {'engine_a': [1, 0.5, ...], ...}
    engine_scores = {name: [] for name in target_engines}
    
    with Pool(processes=NUM_PARALLEL_GAMES) as pool:
        # map() distributes tasks and returns a list of lists of results
        all_results = pool.map(process_game, tasks)
    
    # Aggregate results from all games
    for game_results in all_results:
        for engine_name, score in game_results:
            engine_scores[engine_name].append(score)

    # Calculate average score for each engine
    avg_scores = {}
    for name, scores in engine_scores.items():
        if scores:
            avg_scores[name] = sum(scores) / len(scores)
        else:
            avg_scores[name] = 0

    # Create a DataFrame from the results
    results_df = pd.DataFrame(list(avg_scores.items()), columns=['engine_name', 'average_hit_score'])
    
    # Merge with the original engine ratings
    final_df = pd.merge(engines_df, results_df, on='engine_name')
    final_df.dropna(subset=['rating', 'average_hit_score'], inplace=True)
    
    if len(final_df) < 2:
        print("Not enough data points to calculate R-squared. Skipping.")
        return 0.0 # Cannot compute R² with less than 2 points

    # --- Calculate R-squared ---
    lin_reg_result = linregress(final_df['rating'], final_df['average_hit_score'])
    r_squared = lin_reg_result.rvalue ** 2
    
    print(f"Finished run. R-squared value: {r_squared:.4f}")
    return r_squared


def main_tuner():
    """
    Main function to orchestrate the R-squared optimization process.
    """
    print("--- Starting R-squared Optimization Tuning Script ---")
    
    # --- Load Engine and Game Data ---
    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
        # Extract numerical rating from engine name if a 'rating' column doesn't exist
        if 'rating' not in engines_df.columns and 'engine_name' in engines_df.columns:
             engines_df['rating'] = pd.to_numeric(engines_df['engine_name'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
             engines_df.dropna(subset=['rating'], inplace=True)
        print(f"Loaded {len(engines_df)} engines from '{ENGINES_CSV_PATH}'.")
    except Exception as e:
        print(f"Error loading or parsing '{ENGINES_CSV_PATH}': {e}", file=sys.stderr)
        return

    print(f"Loading games from '{PGN_FILE_PATH}'...")
    games = []
    with open(PGN_FILE_PATH) as pgn:
        # ... (game loading logic remains the same)
        while True:
            if MAX_GAMES_TO_PROCESS and len(games) >= MAX_GAMES_TO_PROCESS: break
            try:
                game = chess.pgn.read_game(pgn)
                if game is None: break
                games.append(game)
            except Exception: continue
    print(f"Loaded {len(games)} games.")

    if not games:
        print("No games loaded. Exiting.")
        return

    all_results = []
    start_time = time.time()

    # --- Main Tuning Loop ---
    for param_set in PARAMETER_SETS_TO_TEST:
        run_start_time = time.time()
        r_squared_value = run_analysis_for_parameters(games, engines_df, param_set)
        run_duration = time.time() - run_start_time
        
        result = {
            "run_name": param_set["name"],
            "r_squared": r_squared_value,
            "num_moves": param_set["num_moves"],
            "weights": str(param_set["weights"]),
            "duration_seconds": run_duration
        }
        all_results.append(result)

    # --- Final Results ---
    print("\n\n" + "=" * 60)
    print("--- Tuning Complete ---")
    print(f"Total time elapsed: {(time.time() - start_time) / 60:.2f} minutes")
    print("=" * 60)

    if not all_results:
        print("No results were generated.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.sort_values(by="r_squared", ascending=False, inplace=True)
    
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nDetailed tuning results saved to '{OUTPUT_CSV_PATH}'")

    print("\n--- Best Parameter Sets (Optimized for R-squared) ---")
    print(results_df.to_string(index=False))
    print("\nScript finished.")


if __name__ == "__main__":
    main_tuner()
