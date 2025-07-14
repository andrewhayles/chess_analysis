# chess_analyzer_multi_engine_v25_ccrl.py
# A script to analyze PGN chess games and estimate a specific player's rating.
#
# KEY CHANGE (from v24):
#   1. CCRL STANDARDS IMPLEMENTED: The script now incorporates key engine settings
#      used by the Computer Chess Rating List (CCRL) to improve the credibility
#      and consistency of the analysis. This includes uniform Hash size, disabled
#      Ponder and Learning, and support for Syzygy endgame tablebases.
#   2. NEW CONFIGURATION SECTION: Added clear variables at the top to manage
#      these new CCRL-inspired settings.

import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg') # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import logging
import multiprocessing
import json # For saving and loading session data
import sys
import csv
from collections import defaultdict
from functools import partial

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
ENGINES_CSV_PATH = PROJECT_FOLDER / "real_engines.csv"
SESSION_FOLDER = PROJECT_FOLDER / "chess_analysis_session"
SESSION_FOLDER.mkdir(exist_ok=True)

PLAYER_TO_ANALYZE = "Desjardins373"
NUM_ANALYSIS_CORES = 2
NUM_ORACLE_ENGINES = 1

# --- CCRL-INSPIRED ENGINE SETTINGS ---
# These settings ensure a more consistent and credible analysis.
HASH_SIZE_MB = 256 # Set a uniform hash size (in MB) for all engines.
# *** IMPORTANT: UPDATE THIS PATH ***
# Set the full path to the folder containing your Syzygy tablebases (.rtbw, .rtbz).
# If you don't have tablebases, leave this as "" to disable them.
TABLEBASE_PATH = r"C:\path\to\your\syzygy\bases" 

# --- HYBRID ANALYSIS CONTROLS ---
ORACLE_ANALYSIS_DEPTH = 20
ORACLE_TIMEOUT = 300 # (in seconds)
MODEL_BUILD_TIME = 6 # (in seconds)
PLAYER_ANALYSIS_TIME = 6 # (in seconds)

# --- SCRIPT SETUP ---
SESSION_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(SESSION_FOLDER / 'analysis_log.txt'), logging.StreamHandler(sys.stdout)]
)
progress_logger = logging.getLogger('progress_logger')
progress_logger.setLevel(logging.INFO)
progress_logger.propagate = False
progress_file_handler = logging.FileHandler(SESSION_FOLDER / 'progress_summary.txt')
progress_file_handler.setFormatter(logging.Formatter('%(message)s'))
if progress_logger.hasHandlers():
    progress_logger.handlers.clear()
progress_logger.addHandler(progress_file_handler)

if not ENGINES_CSV_PATH.is_file():
    logging.error(f"FATAL: Engines CSV file not found at '{ENGINES_CSV_PATH}'")
    sys.exit()

# --- HELPER FUNCTIONS ---

def get_standard_engine_options(csv_options_str):
    """Merges standard CCRL settings with engine-specific options from the CSV."""
    # Start with our standard, credible testing parameters
    options = {
        "Ponder": False,
        "Hash": HASH_SIZE_MB,
        "OwnBook": False,
        # Add other standard options here if needed, e.g., "Learning": False
    }
    # Add tablebase path only if it's valid
    tb_path = Path(TABLEBASE_PATH)
    if tb_path.is_dir():
        options["SyzygyPath"] = TABLEBASE_PATH
    
    # Load and merge options from the CSV file
    if csv_options_str:
        try:
            csv_options = json.loads(csv_options_str)
            options.update(csv_options)
        except json.JSONDecodeError:
            logging.warning(f"Could not decode JSON options from CSV: {csv_options_str}")
            
    return options

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """Call in a loop to create terminal progress bar"""
    if total == 0: total = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: sys.stdout.write('\n'); sys.stdout.flush()

def calculate_move_quality(centipawn_loss):
    """Converts centipawn loss to a quality score between 0.0 and 1.0."""
    if not isinstance(centipawn_loss, (int, float)) or centipawn_loss is None: return 0.0
    if centipawn_loss <= 10: return 1.0
    if centipawn_loss >= 150: return 0.0
    return 1.0 - ((centipawn_loss - 10) / 140.0)

# --- ISOLATED WORKER FUNCTIONS ---

def oracle_move_worker(args):
    """Worker for a single oracle analysis task. Returns the evaluation score."""
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

def model_move_worker(args, engine_info, ground_truth_evals):
    """Worker for a single model-building move. Returns one quality score."""
    fen, analysis_time = args
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            options = get_standard_engine_options(engine_info.get('uci_options', '{}'))
            engine.configure(options)
            board = chess.Board(fen)
            player_to_move = board.turn
            result = engine.play(board, chess.engine.Limit(time=analysis_time))
            if not result.move: return None
            board.push(result.move)
            info = engine.analyse(board, chess.engine.Limit(time=analysis_time), info=chess.engine.INFO_SCORE)
            if isinstance(info, list): info = info[0]
            if 'score' in info:
                played_move_score = info['score'].pov(player_to_move).score(mate_score=30000)
                best_score_pov = ground_truth_evals[fen] if player_to_move == chess.WHITE else -ground_truth_evals[fen]
                cpl = max(0, best_score_pov - played_move_score)
                return calculate_move_quality(cpl)
    except Exception as e:
        logging.error(f"Model worker for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
    return None

def player_move_worker(args, engine_info, ground_truth_evals):
    """Worker for a single player move. Returns one quality score."""
    fen, move, analysis_time = args
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            options = get_standard_engine_options(engine_info.get('uci_options', '{}'))
            engine.configure(options)
            board = chess.Board(fen)
            player_to_move = board.turn
            board.push(move)
            info = engine.analyse(board, chess.engine.Limit(time=analysis_time), info=chess.engine.INFO_SCORE)
            if isinstance(info, list): info = info[0]
            if 'score' in info:
                played_move_score = info['score'].pov(player_to_move).score(mate_score=30000)
                best_score_pov = ground_truth_evals[fen] if player_to_move == chess.WHITE else -ground_truth_evals[fen]
                cpl = max(0, best_score_pov - played_move_score)
                return calculate_move_quality(cpl)
    except Exception as e:
        logging.error(f"Player analysis worker failed on FEN {fen}: {repr(e)}")
    return None

# --- MAIN ANALYSIS FUNCTIONS ---

def get_averaged_ground_truth(game, oracle_engines, session_data, pgn_path):
    """Gets an averaged evaluation for each position, loading from session if possible."""
    in_progress_data = session_data.get('in_progress_game', {})
    if in_progress_data.get('offset') == game.offset and 'ground_truth_evals' in in_progress_data:
        logging.info("Loading pre-calculated ground truth from session file.")
        return in_progress_data['ground_truth_evals']

    logging.info(f"Generating averaged ground truth with {len(oracle_engines)} oracle engines...")
    fens_to_analyze = [board.fen() for board in game.mainline() if board.turn == (chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK)]
    if not fens_to_analyze: return {}

    position_evals = defaultdict(list)
    for engine_info in oracle_engines:
        tasks = [(fen, engine_info, ORACLE_ANALYSIS_DEPTH, ORACLE_TIMEOUT) for fen in fens_to_analyze]
        print_progress_bar(0, len(tasks), prefix=f'Ground Truth ({engine_info["name"]}):', suffix='Complete', length=50)
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            for i, score in enumerate(pool.imap_unordered(oracle_move_worker, tasks)):
                if score is not None: position_evals[tasks[i][0]].append(score)
                print_progress_bar(i + 1, len(tasks), prefix=f'Ground Truth ({engine_info["name"]}):', suffix='Complete', length=50)

    averaged_evals = {fen: np.mean(scores) for fen, scores in position_evals.items() if scores}
    session_data['in_progress_game']['ground_truth_evals'] = averaged_evals
    save_session(session_data, pgn_path)
    return averaged_evals

def build_model_with_real_engines(ground_truth_evals, model_engines, session_data, pgn_path):
    """Builds the performance model by sequentially processing each engine strength."""
    logging.info(f"Building performance model with {len(model_engines)} real engines...")
    fens_for_model = list(ground_truth_evals.keys())
    if not fens_for_model: return None, None, None, None, None, None

    engine_qualities = session_data.get('in_progress_game', {}).get('model_engine_results', {})
    for i, engine_info in enumerate(model_engines):
        if engine_info['name'] in engine_qualities:
            logging.info(f"Skipping already completed engine: {engine_info['name']}")
            continue
        logging.info(f"Processing model engine {i+1}/{len(model_engines)}: {engine_info['name']}")
        tasks = [(fen, MODEL_BUILD_TIME) for fen in fens_for_model]
        worker_func = partial(model_move_worker, engine_info=engine_info, ground_truth_evals=ground_truth_evals)
        qualities_list = []
        print_progress_bar(0, len(tasks), prefix=f'Building ({engine_info["name"]}):', suffix='Complete', length=50)
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            for j, quality in enumerate(pool.imap_unordered(worker_func, tasks)):
                if quality is not None: qualities_list.append(quality)
                print_progress_bar(j + 1, len(tasks), prefix=f'Building ({engine_info["name"]}):', suffix='Complete', length=50)
        if qualities_list:
            engine_qualities[engine_info['name']] = np.mean(qualities_list)
            session_data['in_progress_game']['model_engine_results'] = engine_qualities
            save_session(session_data, pgn_path)

    ratings_data, quality_scores_data = [], []
    for eng_info in model_engines:
        if eng_info['name'] in engine_qualities:
            ratings_data.append(eng_info['rating'])
            quality_scores_data.append(engine_qualities[eng_info['name']])

    if len(ratings_data) < 3: return None, None, None, None, None, None

    ratings = np.array(ratings_data).reshape(-1, 1)
    quality_scores = np.array(quality_scores_data)
    models = {
        'Linear': {'model': LinearRegression().fit(ratings, quality_scores)},
        'Polynomial': {'model': make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(ratings, quality_scores)}
    }
    for name in models: models[name]['r2'] = models[name]['model'].score(ratings, quality_scores)
    best_model_name = max(models, key=lambda name: models[name]['r2'])
    logging.info(f"Model selected: {best_model_name}. R-squared: {models[best_model_name]['r2']:.4f}")
    return models[best_model_name]['model'], models[best_model_name]['r2'], ratings, quality_scores, best_model_name, engine_qualities

def analyze_player_quality(game, ground_truth_evals, analysis_engine_info):
    """Calculates the average move quality for the target player."""
    logging.info(f"Analyzing player moves...")
    player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    tasks = [(b.fen(), m, PLAYER_ANALYSIS_TIME) for b, m in zip(game.mainline(), game.mainline_moves()) if b.turn == player_color]
    if not tasks: return 0.0, 0
    
    qualities = []
    worker_func = partial(player_move_worker, engine_info=analysis_engine_info, ground_truth_evals=ground_truth_evals)
    print_progress_bar(0, len(tasks), prefix='Player Analysis:', suffix='Complete', length=50)
    with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
        for i, quality in enumerate(pool.imap_unordered(worker_func, tasks)):
            if quality is not None: qualities.append(quality)
            print_progress_bar(i + 1, len(tasks), prefix='Player Analysis:', suffix='Complete', length=50)
    return (np.mean(qualities) if qualities else 0.0), len(qualities)

def estimate_rating_from_quality(model, quality_score):
    """Estimates Elo rating from a quality score using the chosen model."""
    search_ratings = np.arange(800, 3500, 1)
    predicted_qualities = model.predict(search_ratings.reshape(-1, 1))
    return int(search_ratings[np.argmin(np.abs(predicted_qualities - quality_score))])

def save_session(session_data, pgn_path):
    """Saves the current session progress to the fixed session folder."""
    with open(SESSION_FOLDER / pgn_path.with_suffix('.session.json').name, 'w') as f:
        json.dump(session_data, f, indent=4)
    logging.info(f"Session progress saved to {SESSION_FOLDER / pgn_path.with_suffix('.session.json').name}")

def load_session(pgn_path):
    """Loads a previous session's progress from the fixed session folder."""
    session_file = SESSION_FOLDER / pgn_path.with_suffix('.session.json').name
    if session_file.exists():
        logging.info(f"Found existing session file: {session_file}, resuming analysis.")
        try:
            with open(session_file, 'r') as f: return json.load(f)
        except json.JSONDecodeError: return None
    return None

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

    session_data = load_session(pgn_path)
    if not session_data:
        session_data = {'pgn_file': str(pgn_path), 'completed_games_data': [], 'in_progress_game': {}}
        with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
            offsets = []
            while True:
                offset = pgn_file.tell()
                headers = chess.pgn.read_headers(pgn_file)
                if headers is None: break
                if PLAYER_TO_ANALYZE.lower() in (headers.get("White", "?").lower(), headers.get("Black", "?").lower()):
                    offsets.append(offset)
            session_data['games_to_process_indices'] = offsets
        save_session(session_data, pgn_path)

    logging.info(f"Found {len(session_data['games_to_process_indices'])} games remaining to analyze.")
    progress_logger.info(f"Chess Analysis Report for {PLAYER_TO_ANALYZE} from '{pgn_path.name}'")
    progress_logger.info("-" * 50)

    continuous_mode = False
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        while session_data['games_to_process_indices']:
            offset = session_data['games_to_process_indices'][0]
            pgn_file.seek(offset)
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                session_data['games_to_process_indices'].pop(0); continue
            game.offset = offset
            if session_data.get('in_progress_game', {}).get('offset') != offset:
                session_data['in_progress_game'] = {'offset': offset}

            white, black = game.headers.get('White', '?'), game.headers.get('Black', '?')
            game_num = len(session_data['completed_games_data']) + 1
            logging.info(f"--- Starting Analysis for Game {game_num}: {white} vs. {black} ---")
            
            ground_truth_evals = get_averaged_ground_truth(game, oracle_engines, session_data, pgn_path)
            if not ground_truth_evals:
                session_data['games_to_process_indices'].pop(0); save_session(session_data, pgn_path); continue

            model_results = build_model_with_real_engines(ground_truth_evals, model_engines, session_data, pgn_path)
            if model_results[0] is None:
                session_data['games_to_process_indices'].pop(0); save_session(session_data, pgn_path); continue
            
            model, r_squared, ratings, quality_scores, model_type, raw_model_results = model_results
            
            avg_quality, moves_counted = analyze_player_quality(game, ground_truth_evals, oracle_engines[0])
            if moves_counted == 0:
                session_data['games_to_process_indices'].pop(0); save_session(session_data, pgn_path); continue

            est_rating = estimate_rating_from_quality(model, avg_quality)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(ratings.flatten(), quality_scores, alpha=0.7, label="Engine Performance")
            plot_x = np.linspace(ratings.min(), ratings.max(), 200).reshape(-1, 1)
            plt.plot(plot_x, model.predict(plot_x), color='red', lw=2, label=f"{model_type} Model (R²={r_squared:.4f})")
            plt.axhline(y=avg_quality, color='g', linestyle='--', label=f"Player Quality ({avg_quality:.4f}) -> {est_rating} Elo")
            plt.title(f"Game {game_num}: {white} vs. {black}"); plt.xlabel("Engine Elo Rating"); plt.ylabel("Move Quality"); plt.grid(True); plt.legend()
            graph_path = SESSION_FOLDER / f"performance_graph_game_{game_num}.png"
            plt.savefig(graph_path); plt.close()
            
            progress_logger.info(f"Game {game_num}: {white} vs. {black} -> Est. Rating: {est_rating} (Model: {model_type}, R²={r_squared:.3f}, Moves: {moves_counted})")

            session_data['completed_games_data'].append({
                'game_num': game_num, 'white': white, 'black': black, 'r_squared': r_squared,
                'graph_path': str(graph_path), 'model_type': model_type, 'avg_quality': avg_quality,
                'moves': moves_counted, 'estimated_rating': est_rating, 'model_results_raw': raw_model_results
            })
            session_data['games_to_process_indices'].pop(0)
            session_data['in_progress_game'] = {}
            save_session(session_data, pgn_path)
            logging.info(f"--- Finished Analysis for Game {game_num}. Estimated rating: {est_rating}. ---")

            if not continuous_mode and session_data['games_to_process_indices']:
                user_input = input("Continue with next game? (y/n/c for yes/no/continuous): ").lower().strip()
                if user_input == 'n': break
                if user_input == 'c': continuous_mode = True

    logging.info("All games analyzed.")
    completed_games = session_data.get('completed_games_data', [])
    if completed_games:
        final_avg_rating = int(np.mean([g['estimated_rating'] for g in completed_games]))
        summary = f"\nFinal Average Estimated Rating: {final_avg_rating} Elo across {len(completed_games)} games."
        progress_logger.info("-" * 50); progress_logger.info(summary)
        logging.info(summary)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
