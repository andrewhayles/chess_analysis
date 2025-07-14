import chess
import chess.pgn
import chess.engine
import pandas as pd
import time
import math
import sys
import os
import multiprocessing
import tqdm
from collections import defaultdict
import itertools
from scipy import stats

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- Parallelism ---
NUM_CORES = 2 # Number of CPU cores to use.

# --- Test Suite Generation ---
# The number of unique positions to pick from the PGN to form the test suite.
NUM_TEST_POSITIONS = 5
# The move number to start considering positions from.
START_MOVE = 10

# --- Engine & File Paths ---
# The 'real_engines.csv' file MUST contain 'engine_name' and 'rating' columns.
ENGINES_CSV_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/real_engines.csv"
PGN_FILE_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/TCEC games/TCEC.pgn"
# The script will automatically identify the highest-rated engine as the Oracle.
# You can override it here if you want (e.g., ORACLE_ENGINE_NAME = "Stockfish 15").
# Set to None to use the auto-detection.
ORACLE_ENGINE_NAME = None
# Path to the engine executables. The script assumes they are in the current
# directory or in the system's PATH. If not, you may need to specify full paths.
# For this script, we assume engine names in the CSV match their executable names.

# --- OPTIMIZATION SETTINGS ---
# Define the search space for the CAI constants.
# The script will test every combination of these values.
# WARNING: The total number of combinations is len(K_WP) * len(K3) * len(W).
# Keep these lists small initially!
K_WP_RANGE = [0.003, 0.004, 0.005]
K3_CRITICALITY_RANGE = [0.002, 0.003, 0.004]
W_IMPACT_RANGE = [0.8, 1.0, 1.2]

# --- Analysis Settings ---
# Analysis depth for all engines.
ANALYSIS_DEPTH = 22 # Reduced depth for faster optimization runs.

# ==============================================================================
# --- Helper & Worker Functions ---
# ==============================================================================

def cp_to_wp(score_obj, k_wp):
    """Converts a centipawn score to Win Probability using a given k_wp."""
    if score_obj.is_mate():
        return 100.0 if score_obj.mate() > 0 else 0.0
    cp = score_obj.score()
    if cp is None: return 50.0
    return 50.0 + 50.0 * math.tanh(k_wp * cp / 2.0)

def calculate_cai_for_move(oracle_engine, board, test_engine_move, oracle_best_move_analysis, constants):
    """
    Calculates the CAI score for a single move made by a test engine.
    The score is relative to the Oracle's assessment.
    """
    k_wp, k3, w = constants
    
    # 1. Get WP for the Oracle's best move (this is our baseline).
    oracle_best_pov_score = oracle_best_move_analysis[0]['score'].pov(board.turn)
    wp_before_best_move = cp_to_wp(oracle_best_pov_score, k_wp)

    # 2. Get WP after the test engine's move is made.
    board.push(test_engine_move)
    info_after_test_move = oracle_engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
    pov_score_after_test_move = info_after_test_move['score'].pov(not board.turn)
    wp_after_test_move = cp_to_wp(pov_score_after_test_move, k_wp)
    board.pop()

    # 3. Calculate Win Probability Loss (WPLoss).
    wp_loss = wp_before_best_move - wp_after_test_move

    # 4. Calculate Criticality based on the Oracle's top 2 moves.
    if len(oracle_best_move_analysis) < 2:
        criticality = 0
    else:
        cp1 = oracle_best_move_analysis[0]['score'].score(mate_score=32000)
        cp2 = oracle_best_move_analysis[1]['score'].score(mate_score=32000)
        delta = abs(cp1 - cp2) if cp1 is not None and cp2 is not None else 0
        criticality = math.tanh(k3 * delta / 2.0)
        
    # 5. Calculate final Impact score.
    impact = wp_loss * (1 + w * criticality)
    return impact

def analyze_position_worker(job):
    """
    Worker function for multiprocessing.
    - Initializes Oracle and Test engines.
    - Has the Test engine choose a move for a given position.
    - Calculates the CAI score for that move against the Oracle's baseline.
    """
    fen, test_engine_name, oracle_engine_name, constants = job
    oracle_engine, test_engine = None, None
    try:
        # Each worker must initialize its own engine instances.
        oracle_engine = chess.engine.SimpleEngine.popen_uci(oracle_engine_name)
        test_engine = chess.engine.SimpleEngine.popen_uci(test_engine_name)
        
        board = chess.Board(fen)
        
        # Let the test engine decide on its move.
        test_engine_result = test_engine.play(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
        test_engine_move = test_engine_result.move

        # Let the Oracle analyze the position to find the best move and criticality.
        oracle_analysis = oracle_engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH), multipv=2)
        
        if not oracle_analysis:
            return None

        cai_score = calculate_cai_for_move(oracle_engine, board, test_engine_move, oracle_analysis, constants)
        
        return (test_engine_name, cai_score)

    except Exception as e:
        # Silently fail on engine errors for robustness, or print for debugging.
        # print(f"Worker error for {test_engine_name} on {fen}: {e}", file=sys.stderr)
        return None
    finally:
        if oracle_engine: oracle_engine.quit()
        if test_engine: test_engine.quit()

# ==============================================================================
# --- Main Optimization Function ---
# ==============================================================================

def optimize_and_compare():
    """
    Main function to orchestrate the CAI constant optimization.
    """
    print("--- Starting CAI Metric Optimization Script ---")

    # --- 1. Load Engine Data and Identify Oracle ---
    try:
        df = pd.read_csv(ENGINES_CSV_PATH)
        if 'rating' not in df.columns:
            print(f"Error: '{ENGINES_CSV_PATH}' must have 'engine_name' and 'rating' columns.", file=sys.stderr)
            return
        df = df.sort_values('rating', ascending=False).reset_index()
        engine_data = {row['engine_name']: {'rating': row['rating']} for index, row in df.iterrows()}
    except Exception as e:
        print(f"Failed to read or parse '{ENGINES_CSV_PATH}': {e}", file=sys.stderr)
        return

    oracle_name = ORACLE_ENGINE_NAME if ORACLE_ENGINE_NAME else df.iloc[0]['engine_name']
    # CORRECTED: Test ALL engines, including the Oracle itself.
    engines_to_test = list(engine_data.keys())
    
    print(f"Oracle Engine: {oracle_name} (Rating: {engine_data[oracle_name]['rating']})")
    print(f"Engines to Test: {len(engines_to_test)} (Including the Oracle)")

    # --- 2. Build Test Suite from PGN ---
    print(f"\nScanning PGN to build a test suite of {NUM_TEST_POSITIONS} positions...")
    test_suite_fens = []
    with open(PGN_FILE_PATH) as pgn:
        while len(test_suite_fens) < NUM_TEST_POSITIONS:
            try:
                game = chess.pgn.read_game(pgn)
                if game is None: break
                board = game.board()
                for i, move in enumerate(game.mainline_moves()):
                    if (i // 2) + 1 >= START_MOVE:
                        test_suite_fens.append(board.fen())
                        if len(test_suite_fens) >= NUM_TEST_POSITIONS: break
                    board.push(move)
            except Exception:
                continue
    if len(test_suite_fens) < NUM_TEST_POSITIONS:
        print(f"Warning: Only found {len(test_suite_fens)} positions meeting criteria.")
    print(f"Test suite created with {len(test_suite_fens)} positions.")

    # --- 3. Run Optimization Loop ---
    constant_combinations = list(itertools.product(K_WP_RANGE, K3_CRITICALITY_RANGE, W_IMPACT_RANGE))
    print(f"\nStarting optimization. Testing {len(constant_combinations)} constant combinations...")
    
    best_r2 = -2.0 # Start with a value lower than -1
    best_constants = {}
    
    for i, constants in enumerate(constant_combinations):
        k_wp, k3, w = constants
        print(f"\n--- Testing Combo {i+1}/{len(constant_combinations)}: K_WP={k_wp}, K3={k3}, W={w} ---")
        
        # Create jobs for this specific combination
        jobs = []
        for fen in test_suite_fens:
            for engine_name in engines_to_test:
                jobs.append((fen, engine_name, oracle_name, constants))

        # Run analysis in parallel
        all_results = []
        with multiprocessing.Pool(processes=NUM_CORES) as pool:
            results_iterator = pool.imap_unordered(analyze_position_worker, jobs)
            for result in tqdm.tqdm(results_iterator, total=len(jobs), desc="Analyzing"):
                if result:
                    all_results.append(result)
        
        # Aggregate results
        engine_scores = defaultdict(list)
        for name, score in all_results:
            engine_scores[name].append(score)
            
        avg_scores = {name: sum(scores) / len(scores) for name, scores in engine_scores.items() if scores}
        
        # Calculate R-squared
        if len(avg_scores) < 2:
            print("Not enough data to calculate R-squared. Skipping.")
            continue

        ratings = [engine_data[name]['rating'] for name in avg_scores.keys()]
        cais = [avg_scores[name] for name in avg_scores.keys()]
        
        try:
            lin_regress = stats.linregress(ratings, cais)
            current_r2 = lin_regress.rvalue ** 2
            print(f"Result: R-squared = {current_r2:.6f}")
            
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_constants = {'K_WP': k_wp, 'K3': k3, 'W': w}
                print(f"*** New best R-squared found! ***")
        except ValueError as e:
            print(f"Could not calculate R-squared: {e}")


    # --- 4. Final Report ---
    print("\n\n" + "=" * 60)
    print("--- Optimization Complete ---")
    print("=" * 60)
    if not best_constants:
        print("No successful analysis runs were completed.")
    else:
        print(f"Highest R-squared value achieved: {best_r2:.6f}")
        print("Best performing constants:")
        print(f"  - K_WP (for Win Probability): {best_constants['K_WP']}")
        print(f"  - K3 (for Criticality):      {best_constants['K3']}")
        print(f"  - W (for Impact Weighting):  {best_constants['W']}")
    print("=" * 60)
    print("\nScript finished.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    optimize_and_compare()
