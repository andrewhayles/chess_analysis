# optimizer_v1.py
# A specialized script to find the optimal analysis time for the main rating script.
#
# HOW IT WORKS:
# 1. It analyzes ONLY THE FIRST GAME of the provided PGN file.
# 2. It generates a high-quality "ground truth" for this game one time.
# 3. It then iterates through a list of different timeout values.
# 4. For each timeout, it builds a complete performance model and calculates its R-squared value.
# 5. Finally, it reports which timeout setting produced the model with the best fit (highest R-squared).

import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from pathlib import Path
import datetime
import logging
import multiprocessing
import json
import sys
import csv
from collections import defaultdict
from functools import partial

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
ENGINES_CSV_PATH = PROJECT_FOLDER / "real_engines.csv"
SESSION_FOLDER = PROJECT_FOLDER / "chess_analysis_session_optimizer"
SESSION_FOLDER.mkdir(exist_ok=True)

# --- OPTIMIZATION SETTINGS ---
# Define the different timeout values (in seconds) to test.
TIMEOUTS_TO_TEST = [2, 6, 15, 30, 45, 60, 90, 105, 120, 180, 240]
# Define the fixed depth for this optimization test.
OPTIMIZATION_DEPTH = 12

# --- FIXED ORACLE SETTINGS ---
ORACLE_ANALYSIS_DEPTH = 20
ORACLE_TIMEOUT = 600

# --- GENERAL SETTINGS ---
PLAYER_TO_ANALYZE = "Desjardins373"
NUM_ANALYSIS_CORES = 2
NUM_ORACLE_ENGINES = 1
HASH_SIZE_MB = 256
TABLEBASE_PATH = r""

# --- SCRIPT SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(SESSION_FOLDER / 'optimizer_log.txt'), logging.StreamHandler(sys.stdout)]
)

if not ENGINES_CSV_PATH.is_file():
    logging.error(f"FATAL: Engines CSV file not found at '{ENGINES_CSV_PATH}'")
    sys.exit()

# --- HELPER & WORKER FUNCTIONS (Copied from main script) ---

def get_standard_engine_options(csv_options_str):
    options = {"Hash": HASH_SIZE_MB}
    tb_path = Path(TABLEBASE_PATH)
    if tb_path.is_dir() and tb_path.exists():
        options["SyzygyPath"] = str(tb_path)
    if csv_options_str:
        try:
            options.update(json.loads(csv_options_str))
        except json.JSONDecodeError:
            logging.warning(f"Could not decode JSON options from CSV: {csv_options_str}")
    return options

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    if total == 0: total = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: sys.stdout.write('\n'); sys.stdout.flush()

def calculate_move_quality(centipawn_loss):
    if not isinstance(centipawn_loss, (int, float)) or centipawn_loss is None: return 0.0
    if centipawn_loss <= 10: return 1.0
    if centipawn_loss >= 150: return 0.0
    return 1.0 - ((centipawn_loss - 10) / 140.0)

def oracle_worker(args):
    fen, engine_info, depth, timeout = args
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            options = get_standard_engine_options(engine_info.get('uci_options', '{}'))
            engine.configure(options)
            info = engine.analyse(chess.Board(fen), chess.engine.Limit(depth=depth, time=timeout), multipv=1)
            if isinstance(info, list): info = info[0]
            if 'score' in info:
                return info['score'].white().score(mate_score=30000)
    except Exception as e:
        logging.error(f"Oracle worker for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
    return None

def move_analysis_worker(args):
    fen, move, depth, timeout, engine_info, ground_truth_evals = args
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            options = get_standard_engine_options(engine_info.get('uci_options', '{}'))
            engine.configure(options)
            board = chess.Board(fen)
            player_to_move = board.turn
            if move is None:
                result = engine.play(board, chess.engine.Limit(depth=depth, time=timeout))
                if not result.move: return None
                board.push(result.move)
            else:
                board.push(move)
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=timeout), info=chess.engine.INFO_SCORE)
            if isinstance(info, list): info = info[0]
            if 'score' in info:
                played_move_score = info['score'].pov(player_to_move).score(mate_score=30000)
                best_score_pov = ground_truth_evals[fen] if player_to_move == chess.WHITE else -ground_truth_evals[fen]
                cpl = max(0, best_score_pov - played_move_score)
                return calculate_move_quality(cpl)
    except Exception as e:
        logging.error(f"Worker for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
    return None

# --- OPTIMIZER-SPECIFIC FUNCTIONS ---

def get_ground_truth_for_one_game(game, oracle_engines):
    logging.info("Generating averaged ground truth for optimization game...")
    player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    fens_to_analyze = [node.board().fen() for node in game.mainline() if node.board().turn == player_color]
    if not fens_to_analyze:
        logging.error("No moves found for player in the first game. Cannot optimize.")
        return None

    position_evals = defaultdict(list)
    for engine_info in oracle_engines:
        tasks = [(fen, engine_info, ORACLE_ANALYSIS_DEPTH, ORACLE_TIMEOUT) for fen in fens_to_analyze]
        print_progress_bar(0, len(tasks), prefix=f'Ground Truth ({engine_info["name"]}):', suffix='Complete', length=50)
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            for i, score in enumerate(pool.imap_unordered(oracle_worker, tasks)):
                if score is not None: position_evals[tasks[i][0]].append(score)
                print_progress_bar(i + 1, len(tasks), prefix=f'Ground Truth ({engine_info["name"]}):', suffix='Complete', length=50)

    return {fen: np.mean(scores) for fen, scores in position_evals.items() if scores}

def test_timeout_for_model(timeout, ground_truth_evals, model_engines):
    logging.info(f"----- TESTING TIMEOUT: {timeout} seconds -----")
    engine_qualities = {}
    for i, engine_info in enumerate(model_engines):
        logging.info(f"  Processing model engine {i+1}/{len(model_engines)}: {engine_info['name']}")
        tasks = [(fen, None, OPTIMIZATION_DEPTH, timeout, engine_info, ground_truth_evals) for fen in ground_truth_evals]
        qualities_list = []
        print_progress_bar(0, len(tasks), prefix=f'  Building ({engine_info["name"]} @ {timeout}s):', suffix='Complete', length=50)
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            for j, quality in enumerate(pool.imap_unordered(move_analysis_worker, tasks)):
                if quality is not None: qualities_list.append(quality)
                print_progress_bar(j + 1, len(tasks), prefix=f'  Building ({engine_info["name"]} @ {timeout}s):', suffix='Complete', length=50)
        if qualities_list:
            engine_qualities[engine_info['name']] = np.mean(qualities_list)

    ratings_data, quality_scores_data = [], []
    for eng_info in model_engines:
        if eng_info['name'] in engine_qualities:
            ratings_data.append(eng_info['rating'])
            quality_scores_data.append(engine_qualities[eng_info['name']])

    if len(ratings_data) < 3: return 0.0
    ratings = np.array(ratings_data).reshape(-1, 1)
    quality_scores = np.array(quality_scores_data)
    
    poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(ratings, quality_scores)
    poly_r2 = poly_model.score(ratings, quality_scores)
    
    linear_model = LinearRegression().fit(ratings, quality_scores)
    linear_r2 = linear_model.score(ratings, quality_scores)
    
    best_r2 = max(poly_r2, linear_r2)
    logging.info(f"----- Result for {timeout}s -> R-squared: {best_r2:.4f} -----")
    return best_r2

# --- MAIN EXECUTION LOGIC ---
def main():
    all_engines = []
    try:
        with open(ENGINES_CSV_PATH, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                row['rating'] = int(row['rating'])
                all_engines.append(row)
    except Exception as e:
        logging.error(f"FATAL: Error reading engines CSV: {e}"); return

    oracle_engines, model_engines = all_engines[:NUM_ORACLE_ENGINES], all_engines[NUM_ORACLE_ENGINES:]
    logging.info(f"Loaded {len(oracle_engines)} oracle engines and {len(model_engines)} model-building engines.")

    pgn_path = Path(input("Enter the full path to your PGN file: "))
    if not pgn_path.is_file(): logging.error(f"PGN file not found at {pgn_path}"); return

    first_game = None
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None: break
            if PLAYER_TO_ANALYZE.lower() in (game.headers.get("White", "?").lower(), game.headers.get("Black", "?").lower()):
                first_game = game
                break
    
    if not first_game:
        logging.error(f"No games found for player {PLAYER_TO_ANALYZE} in PGN file.")
        return

    logging.info(f"Found first game for optimization: {first_game.headers.get('White')} vs. {first_game.headers.get('Black')}")

    ground_truth_evals = get_ground_truth_for_one_game(first_game, oracle_engines)
    if not ground_truth_evals:
        logging.error("Failed to generate ground truth. Aborting optimization.")
        return

    optimization_results = {}
    for timeout in TIMEOUTS_TO_TEST:
        r_squared = test_timeout_for_model(timeout, ground_truth_evals, model_engines)
        optimization_results[timeout] = r_squared

    logging.info("\n\n----- OPTIMIZATION COMPLETE -----")
    logging.info("R-squared results for different timeout values:")
    for timeout, r2 in optimization_results.items():
        logging.info(f"  - {timeout} seconds: R² = {r2:.4f}")

    if optimization_results:
        best_timeout = max(optimization_results, key=optimization_results.get)
        best_r2 = optimization_results[best_timeout]
        logging.info(f"\nOptimal Timeout: {best_timeout} seconds (with R² = {best_r2:.4f})")
        logging.info(f"RECOMMENDATION: Update MODEL_BUILD_TIMEOUT and PLAYER_ANALYSIS_TIMEOUT to {best_timeout} in the main script.")
    else:
        logging.warning("No optimization results were generated.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
