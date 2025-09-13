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
import subprocess
import csv

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- Parallelism ---
NUM_CORES = 2 # Number of CPU cores to use.

# --- Test Suite Generation ---
# The number of unique positions to pick from the PGN to form the test suite.
NUM_TEST_POSITIONS = 10
# The move number to start considering positions from.
START_MOVE = 10

# --- Engine & File Paths ---
# The 'real_engines.csv' file MUST contain 'engine_name' and 'rating' columns.
# It can also contain an optional 'executable_name' column for batch files.
ENGINES_CSV_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/real_engines.csv"
PGN_FILE_PATH = "C:/Users/desja/Documents/Python_programs/chess_study/TCEC games/TCEC.pgn"
# Directory where engine executables/batch files are located.
ENGINE_DIR = "C:/Users/desja/Documents/Python_programs/chess_study/engines/leela_chess"
# NEW: Log file for live results and resuming.
LOG_FILE_PATH = "C:/Users/desja/Documents/Python_programs/chess_study//CAI_output/analysis_results.csv"


# --- OPTIMIZATION SETTINGS ---
# Define the search space for the CAI constants.
K_WP_RANGE = [0.003, 0.004, 0.005]
K3_CRITICALITY_RANGE = [0.002, 0.003, 0.004]
W_IMPACT_RANGE = [0.8, 1.0, 1.2]

# --- Analysis Settings ---
# Analysis will stop when EITHER the depth OR the timeout is reached for each engine type.
# Use a deeper/longer setting for the Oracle to get a more accurate "ground truth".
ORACLE_ANALYSIS_DEPTH = 22
ORACLE_TIMEOUT_SECONDS = 600
# Use a shallower/shorter setting for the test engines to simulate match conditions.
TEST_ENGINE_ANALYSIS_DEPTH = 12
TEST_ENGINE_TIMEOUT_SECONDS = 6


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

def calculate_cai_for_move(oracle_engine, board, test_engine_move, oracle_best_move_analysis, constants, oracle_limit):
    """Calculates the CAI score for a single move made by a test engine."""
    k_wp, k3, w = constants
    oracle_best_pov_score = oracle_best_move_analysis[0]['score'].pov(board.turn)
    wp_before_best_move = cp_to_wp(oracle_best_pov_score, k_wp)

    board.push(test_engine_move)
    info_after_test_move = oracle_engine.analyse(board, oracle_limit)
    pov_score_after_test_move = info_after_test_move['score'].pov(not board.turn)
    wp_after_test_move = cp_to_wp(pov_score_after_test_move, k_wp)
    board.pop()

    wp_loss = wp_before_best_move - wp_after_test_move

    if len(oracle_best_move_analysis) < 2:
        criticality = 0
    else:
        pov_score1 = oracle_best_move_analysis[0]['score'].pov(board.turn)
        pov_score2 = oracle_best_move_analysis[1]['score'].pov(board.turn)
        cp1 = pov_score1.score(mate_score=32000)
        cp2 = pov_score2.score(mate_score=32000)
        delta = abs(cp1 - cp2) if cp1 is not None and cp2 is not None else 0
        criticality = math.tanh(k3 * delta / 2.0)
        
    impact = wp_loss * (1 + w * criticality)
    return impact

def analyze_position_worker(job):
    """Worker function with robust cleanup and suppressed engine stderr."""
    fen, test_engine_name, test_engine_executable, oracle_executable, constants, limits = job
    oracle_depth, oracle_timeout, test_depth, test_timeout = limits
    
    oracle_engine, test_engine = None, None
    oracle_path, test_path = None, None
    try:
        oracle_path = os.path.join(ENGINE_DIR, oracle_executable)
        test_path = os.path.join(ENGINE_DIR, test_engine_executable)
        
        oracle_engine = chess.engine.SimpleEngine.popen_uci(oracle_path, stderr=subprocess.DEVNULL)
        test_engine = chess.engine.SimpleEngine.popen_uci(test_path, stderr=subprocess.DEVNULL)
        
        board = chess.Board(fen)
        
        oracle_limit = chess.engine.Limit(depth=oracle_depth, time=oracle_timeout)
        test_engine_limit = chess.engine.Limit(depth=test_depth, time=test_timeout)
        
        test_engine_result = test_engine.play(board, test_engine_limit)
        test_engine_move = test_engine_result.move

        oracle_analysis = oracle_engine.analyse(board, oracle_limit, multipv=2)
        if not oracle_analysis:
            raise Exception("Oracle analysis failed.")

        cai_score = calculate_cai_for_move(oracle_engine, board, test_engine_move, oracle_analysis, constants, oracle_limit)
        # Return constants along with the result for logging
        return ('success', (constants, fen, test_engine_name, cai_score))

    except Exception as e:
        error_msg = str(e).replace('\\\\', '\\')
        return ('error', (test_engine_name, error_msg, test_path))
    finally:
        if oracle_engine:
            try: oracle_engine.quit()
            except chess.engine.EngineTerminatedError: pass 
        if test_engine:
            try: test_engine.quit()
            except chess.engine.EngineTerminatedError: pass

def load_completed_jobs(log_file):
    """Reads the log file to find jobs that have already been completed."""
    if not os.path.exists(log_file):
        return set()
    
    completed = set()
    with open(log_file, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            # Find indices of columns to build the unique job identifier
            k_wp_idx = header.index('k_wp')
            k3_idx = header.index('k3')
            w_idx = header.index('w')
            fen_idx = header.index('fen')
            engine_idx = header.index('engine_name')
            for row in reader:
                try:
                    job_id = (
                        float(row[k_wp_idx]),
                        float(row[k3_idx]),
                        float(row[w_idx]),
                        row[fen_idx],
                        row[engine_idx]
                    )
                    completed.add(job_id)
                except (ValueError, IndexError):
                    continue # Skip malformed rows
        except (StopIteration, ValueError):
             return set() # File is empty or has no valid header
    return completed

# ==============================================================================
# --- Main Optimization Function ---
# ==============================================================================

def optimize_and_compare():
    """Main function to orchestrate the CAI constant optimization."""
    print("--- Starting CAI Metric Optimization Script ---")

    # --- 1. Load Engine Data and Identify Oracle ---
    try:
        df = pd.read_csv(ENGINES_CSV_PATH, comment='#')
        engine_data = {
            row['engine_name']: {
                'rating': row['rating'],
                'executable': row.get('executable_name', row['engine_name'])
            } for _, row in df.iterrows()
        }
        df = df.sort_values('rating', ascending=False)
    except Exception as e:
        print(f"Failed to read or parse '{ENGINES_CSV_PATH}': {e}", file=sys.stderr)
        return

    oracle_logical_name = df.iloc[0]['engine_name']
    oracle_executable_name = engine_data[oracle_logical_name]['executable']
    engines_to_test = list(engine_data.keys())
    
    print(f"Oracle Engine: {oracle_logical_name} (using executable: {oracle_executable_name})")
    print(f"Engines to Test: {len(engines_to_test)} (Including the Oracle)")

    # --- 2. Build Test Suite from PGN ---
    print(f"\nScanning PGN to build a test suite of {NUM_TEST_POSITIONS} positions...")
    test_suite_fens = []
    with open(PGN_FILE_PATH, 'r', errors='ignore') as pgn:
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
    print(f"Test suite created with {len(test_suite_fens)} positions.")

    # --- 3. Load previous results and create new jobs ---
    print(f"Loading previously completed jobs from '{LOG_FILE_PATH}'...")
    completed_jobs = load_completed_jobs(LOG_FILE_PATH)
    print(f"Found {len(completed_jobs)} completed jobs.")

    all_jobs_to_run = []
    constant_combinations = list(itertools.product(K_WP_RANGE, K3_CRITICALITY_RANGE, W_IMPACT_RANGE))
    limits = (ORACLE_ANALYSIS_DEPTH, ORACLE_TIMEOUT_SECONDS, TEST_ENGINE_ANALYSIS_DEPTH, TEST_ENGINE_TIMEOUT_SECONDS)

    for constants in constant_combinations:
        for fen in test_suite_fens:
            for name in engines_to_test:
                job_id = (*constants, fen, name)
                if job_id not in completed_jobs:
                    executable = engine_data[name]['executable']
                    all_jobs_to_run.append((fen, name, executable, oracle_executable_name, constants, limits))

    if not all_jobs_to_run:
        print("\nAll analysis jobs are already complete.")
    else:
        print(f"\nStarting analysis for {len(all_jobs_to_run)} new jobs...")
        log_file_exists = os.path.exists(LOG_FILE_PATH)
        with open(LOG_FILE_PATH, 'a', newline='', buffering=1) as f:
            writer = csv.writer(f)
            if not log_file_exists or os.path.getsize(LOG_FILE_PATH) == 0:
                writer.writerow(['k_wp', 'k3', 'w', 'fen', 'engine_name', 'cai_score'])
            
            with multiprocessing.Pool(processes=NUM_CORES) as pool:
                results_iterator = pool.imap_unordered(analyze_position_worker, all_jobs_to_run)
                for result in tqdm.tqdm(results_iterator, total=len(all_jobs_to_run), desc="Analyzing"):
                    if result and result[0] == 'success':
                        constants, fen, name, score = result[1]
                        k_wp, k3, w = constants
                        writer.writerow([k_wp, k3, w, fen, name, score])

    # --- 4. Final Report based on the entire log file ---
    print("\n--- Aggregating All Results for Final Report ---")
    try:
        full_df = pd.read_csv(LOG_FILE_PATH)
    except FileNotFoundError:
        print("Log file not found. No results to report.")
        return
        
    best_r2 = -2.0
    best_constants = {}

    for constants, group in full_df.groupby(['k_wp', 'k3', 'w']):
        k_wp, k3, w = constants
        
        # Merge with engine data to get ratings
        group = pd.merge(group, df[['engine_name', 'rating']], on='engine_name', how='left')
        
        if len(group['engine_name'].unique()) < 2:
            continue

        avg_scores = group.groupby('engine_name')['cai_score'].mean()
        avg_scores_df = avg_scores.reset_index()
        
        merged_df = pd.merge(avg_scores_df, df[['engine_name', 'rating']], on='engine_name')

        if len(merged_df) < 2:
            continue
            
        try:
            lin_regress = stats.linregress(merged_df['rating'], merged_df['cai_score'])
            current_r2 = lin_regress.rvalue ** 2
            
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_constants = {'K_WP': k_wp, 'K3': k3, 'W': w}
        except ValueError:
            continue

    print("\n\n" + "=" * 60)
    print("--- Optimization Complete ---")
    print("=" * 60)
    if not best_constants:
        print("No successful analysis runs were completed or R-squared could not be calculated.")
    else:
        print(f"Highest R-squared value achieved: {best_r2:.6f}")
        print("Best performing constants (based on all available data):")
        print(f"  - K_WP (for Win Probability): {best_constants['K_WP']}")
        print(f"  - K3 (for Criticality):      {best_constants['K3']}")
        print(f"  - W (for Impact Weighting):  {best_constants['W']}")
    print("=" * 60)
    print("\nScript finished.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    optimize_and_compare()
