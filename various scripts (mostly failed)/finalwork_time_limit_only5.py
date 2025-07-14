# chess_analyzer_multi_engine_v24_batch_mode.py
# A script to analyze PGN chess games and estimate a specific player's rating.
#
# KEY CHANGE (from v23):
#   1. BATCH/CONTINUOUS MODE RESTORED: The script now runs through all games in the
#      PGN file and prompts the user to continue after each one, with an option
#      for continuous analysis.
#   2. EXTENSIBLE DATA & FOCUSED PARALLELISM RETAINED: This version keeps the
#      advanced data saving format and the sequential engine processing logic.

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

# Use a fixed folder name for seamless resuming
SESSION_FOLDER = PROJECT_FOLDER / "chess_analysis_session"
SESSION_FOLDER.mkdir(exist_ok=True) # Create the folder if it doesn't exist

# --- RESOURCE CONTROLS ---
NUM_ANALYSIS_CORES = 2 # Set to use exactly two cores for analysis

# --- HYBRID ANALYSIS CONTROLS ---
# Oracle uses depth with a time-based safety net for maximum quality.
ORACLE_ANALYSIS_DEPTH = 20
ORACLE_TIMEOUT = 300 # (in seconds)

# Model and Player analysis use a fixed time for speed.
MODEL_BUILD_TIME = 6 # (in seconds)
PLAYER_ANALYSIS_TIME = 6 # (in seconds)


PLAYER_TO_ANALYZE = "Desjardins373"
# Set to the number of reliable engines you want for ground truth.
# For the Stockfish-only method, this should be 1.
NUM_ORACLE_ENGINES = 1

# --- SCRIPT SETUP ---
SESSION_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Main Logger (for detailed debug info) ---
log_file_path = SESSION_FOLDER / 'analysis_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)]
)

# --- Progress Logger (for clean, user-facing summary) ---
progress_log_path = SESSION_FOLDER / 'progress_summary.txt'
progress_logger = logging.getLogger('progress_logger')
progress_logger.setLevel(logging.INFO)
progress_logger.propagate = False
progress_file_handler = logging.FileHandler(progress_log_path)
progress_file_handler.setFormatter(logging.Formatter('%(message)s'))
if progress_logger.hasHandlers():
    progress_logger.handlers.clear()
progress_logger.addHandler(progress_file_handler)


if not ENGINES_CSV_PATH.is_file():
    logging.error(f"FATAL: Engines CSV file not found at '{ENGINES_CSV_PATH}'")
    sys.exit()

# --- UTILITY FUNCTIONS ---
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """Call in a loop to create terminal progress bar"""
    if total == 0:
        total = 1
        iteration = 1

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def calculate_move_quality(centipawn_loss):
    """Converts centipawn loss to a quality score between 0.0 and 1.0."""
    if not isinstance(centipawn_loss, (int, float)) or centipawn_loss is None:
        return 0.0
    if centipawn_loss <= 10: return 1.0
    if centipawn_loss >= 150: return 0.0
    return 1.0 - ((centipawn_loss - 10) / 140.0)

# --- ISOLATED WORKER FUNCTIONS ---

def oracle_move_worker(args):
    """Worker for a single oracle analysis task. Returns the evaluation score."""
    fen, engine_info, depth, timeout = args
    board = chess.Board(fen)
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            if 'uci_options' in engine_info and engine_info['uci_options']:
                options = json.loads(engine_info['uci_options'])
                engine.configure(options)
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=timeout), multipv=1)
            if isinstance(info, list): info = info[0]
            if 'score' in info:
                score_obj = info['score'].white()
                return score_obj.score(mate_score=30000)
    except Exception as e:
        logging.error(f"Oracle worker for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
    return None

def model_move_worker(args, engine_info, ground_truth_evals):
    """Worker for a single model-building move. Returns one quality score."""
    fen, analysis_time = args
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            if 'uci_options' in engine_info and engine_info['uci_options']:
                engine.configure(json.loads(engine_info['uci_options']))
            
            board = chess.Board(fen)
            player_to_move = board.turn
            result = engine.play(board, chess.engine.Limit(time=analysis_time))
            if not result.move: return None

            board.push(result.move)
            info = engine.analyse(board, chess.engine.Limit(time=analysis_time), info=chess.engine.INFO_SCORE)
            if isinstance(info, list): info = info[0]

            if 'score' in info:
                played_move_score_obj = info['score'].pov(player_to_move)
                played_move_score = played_move_score_obj.score(mate_score=30000)
                ground_truth_white_pov = ground_truth_evals[fen]
                best_score_pov = ground_truth_white_pov if player_to_move == chess.WHITE else -ground_truth_white_pov
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
            if 'uci_options' in engine_info and engine_info['uci_options']:
                engine.configure(json.loads(engine_info['uci_options']))

            board = chess.Board(fen)
            player_to_move = board.turn
            board.push(move)
            info = engine.analyse(board, chess.engine.Limit(time=analysis_time), info=chess.engine.INFO_SCORE)
            if isinstance(info, list): info = info[0]
            
            if 'score' in info:
                played_move_score_obj = info['score'].pov(player_to_move)
                played_move_score = played_move_score_obj.score(mate_score=30000)
                ground_truth_white_pov = ground_truth_evals[fen]
                best_score_pov = ground_truth_white_pov if player_to_move == chess.WHITE else -ground_truth_white_pov
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
        progress_logger.info("  -> Found and loaded pre-calculated ground truth.")
        return in_progress_data['ground_truth_evals']

    logging.info(f"Generating averaged ground truth with {len(oracle_engines)} oracle engines...")
    progress_logger.info(f"  -> Generating ground truth...")

    fens_to_analyze = []
    board = game.board()
    try:
        player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    except AttributeError:
        player_color = chess.BLACK

    for move in game.mainline_moves():
        if board.turn == player_color:
            fens_to_analyze.append(board.fen())
        board.push(move)

    if not fens_to_analyze:
        logging.warning("No moves found for the specified player in this game.")
        return {}

    position_evals = defaultdict(list)
    for engine_info in oracle_engines:
        tasks = [(fen, engine_info, ORACLE_ANALYSIS_DEPTH, ORACLE_TIMEOUT) for fen in fens_to_analyze]
        print_progress_bar(0, len(tasks), prefix=f'Ground Truth ({engine_info["name"]}):', suffix='Complete', length=50)
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            for i, score in enumerate(pool.imap_unordered(oracle_move_worker, tasks)):
                if score is not None:
                    # Associate the score with the correct FEN
                    position_evals[tasks[i][0]].append(score)
                print_progress_bar(i + 1, len(tasks), prefix=f'Ground Truth ({engine_info["name"]}):', suffix='Complete', length=50)

    averaged_evals = {fen: np.mean(scores) for fen, scores in position_evals.items() if scores}
    logging.info("Averaged ground truth generation complete.")
    progress_logger.info(f"  -> Ground truth generated for {len(averaged_evals)} positions.")

    session_data['in_progress_game']['ground_truth_evals'] = averaged_evals
    save_session(session_data, pgn_path)

    return averaged_evals

def build_model_with_real_engines(ground_truth_evals, model_engines, session_data, pgn_path):
    """
    Builds the performance model by sequentially processing each engine strength,
    using multiple cores to analyze the moves for that single engine in parallel.
    """
    logging.info(f"Building performance model with {len(model_engines)} real engines...")
    progress_logger.info(f"  -> Building performance model...")

    fens_for_model = list(ground_truth_evals.keys())
    if not fens_for_model:
        logging.error("Cannot build model: Ground truth data is empty.")
        return None, None, None, None, None, None

    engine_qualities = session_data.get('in_progress_game', {}).get('model_engine_results', {})
    
    for i, engine_info in enumerate(model_engines):
        engine_name = engine_info['name']
        if engine_name in engine_qualities:
            logging.info(f"Skipping already completed engine: {engine_name}")
            continue

        logging.info(f"Processing model engine {i+1}/{len(model_engines)}: {engine_name}")
        tasks = [(fen, MODEL_BUILD_TIME) for fen in fens_for_model]
        worker_func = partial(model_move_worker, engine_info=engine_info, ground_truth_evals=ground_truth_evals)
        
        qualities_list = []
        print_progress_bar(0, len(tasks), prefix=f'Building ({engine_name}):', suffix='Complete', length=50)
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            for j, quality in enumerate(pool.imap_unordered(worker_func, tasks)):
                if quality is not None:
                    qualities_list.append(quality)
                print_progress_bar(j + 1, len(tasks), prefix=f'Building ({engine_name}):', suffix='Complete', length=50)
        
        if qualities_list:
            avg_quality = np.mean(qualities_list)
            engine_qualities[engine_name] = avg_quality
            log_msg = f"  -> Model data for engine: {engine_name}. Avg quality: {avg_quality:.4f} ({len(qualities_list)}/{len(fens_for_model)} moves analyzed)"
            logging.info(log_msg)
            progress_logger.info(f"    - Model data gathered for: {engine_name} ({len(qualities_list)} moves)")
            session_data['in_progress_game']['model_engine_results'] = engine_qualities
            save_session(session_data, pgn_path)
        else:
            logging.warning(f"  -> No model data gathered for engine: {engine_name}. It may have failed to start or analyze any moves.")

    ratings_data, quality_scores_data = [], []
    for eng_info in model_engines:
        name = eng_info['name']
        if name in engine_qualities:
            ratings_data.append(eng_info['rating'])
            quality_scores_data.append(engine_qualities[name])

    if len(ratings_data) < 3:
        logging.warning(f"Failed to gather enough data for model. Only got {len(ratings_data)} points.")
        return None, None, None, None, None, None

    ratings = np.array(ratings_data).reshape(-1, 1)
    quality_scores = np.array(quality_scores_data)

    models = {}
    models['Linear'] = {'model': LinearRegression().fit(ratings, quality_scores)}
    models['Linear']['r2'] = models['Linear']['model'].score(ratings, quality_scores)
    models['Polynomial'] = {'model': make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(ratings, quality_scores)}
    models['Polynomial']['r2'] = models['Polynomial']['model'].score(ratings, quality_scores)
    
    best_model_name = max(models, key=lambda name: models[name]['r2'])
    best_model = models[best_model_name]['model']
    best_r2 = models[best_model_name]['r2']

    logging.info(f"Model selected: {best_model_name}. R-squared: {best_r2:.4f}")
    logging.info(f"  (Scores: Linear R²={models['Linear']['r2']:.4f}, Polynomial R²={models['Polynomial']['r2']:.4f})")
    progress_logger.info(f"  -> Model created ({best_model_name}) with R-squared: {best_r2:.4f}")
    
    return best_model, best_r2, ratings, quality_scores, best_model_name, engine_qualities


def analyze_player_quality(game, ground_truth_evals, analysis_engine_info):
    """Calculates the average move quality for the target player."""
    logging.info(f"Analyzing player moves...")
    progress_logger.info(f"  -> Analyzing player moves...")

    white_player = game.headers.get("White", "?").lower()
    black_player = game.headers.get("Black", "?").lower()
    player_name_lower = PLAYER_TO_ANALYZE.lower()
    player_color = chess.WHITE if player_name_lower == white_player else chess.BLACK

    all_moves = list(game.mainline_moves())
    tasks = []
    board = game.board()
    for move in all_moves:
        if board.turn == player_color:
            tasks.append((board.fen(), move, PLAYER_ANALYSIS_TIME))
        board.push(move)

    if not tasks:
        return 0.0, 0

    qualities = []
    worker_func = partial(player_move_worker, engine_info=analysis_engine_info, ground_truth_evals=ground_truth_evals)
    print_progress_bar(0, len(tasks), prefix='Player Analysis:', suffix='Complete', length=50)
    
    with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
        for i, quality in enumerate(pool.imap_unordered(worker_func, tasks)):
            if quality is not None:
                qualities.append(quality)
            print_progress_bar(i + 1, len(tasks), prefix='Player Analysis:', suffix='Complete', length=50)

    avg_quality = np.mean(qualities) if qualities else 0.0
    return avg_quality, len(qualities)

def estimate_rating_from_quality(model, quality_score, model_type):
    """Estimates Elo rating from a quality score using the chosen model."""
    search_ratings = np.arange(800, 3500, 1)
    predicted_qualities = model.predict(search_ratings.reshape(-1, 1))
    closest_rating_index = np.argmin(np.abs(predicted_qualities - quality_score))
    return int(search_ratings[closest_rating_index])


def save_session(session_data, pgn_path):
    """Saves the current session progress to the fixed session folder."""
    session_file = SESSION_FOLDER / pgn_path.with_suffix('.session.json').name
    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=4)
    logging.info(f"Session progress saved to {session_file}")

def load_session(pgn_path):
    """Loads a previous session's progress from the fixed session folder."""
    session_file = SESSION_FOLDER / pgn_path.with_suffix('.session.json').name
    if session_file.exists():
        logging.info(f"Found existing session file: {session_file}, resuming analysis.")
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Could not read session file {session_file}. Starting fresh.")
            return None
    logging.info("No existing session file found. Starting a new analysis.")
    return None

def generate_final_report(session_data):
    logging.info("Generating final PDF report...")
    # This function can be expanded to create a detailed PDF summary.
    pass

# --- MAIN EXECUTION LOGIC ---

def main():
    # --- Load Engines from CSV ---
    all_engines = []
    try:
        with open(ENGINES_CSV_PATH, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                row['rating'] = int(row['rating'])
                all_engines.append(row)
    except Exception as e:
        logging.error(f"FATAL: Error reading engines CSV: {e}")
        return

    if len(all_engines) <= NUM_ORACLE_ENGINES:
        logging.error("FATAL: Not enough engines in CSV for both oracle and model panels.")
        return

    oracle_engines = all_engines[:NUM_ORACLE_ENGINES]
    model_engines = all_engines[NUM_ORACLE_ENGINES:]
    logging.info(f"Loaded {len(oracle_engines)} oracle engines and {len(model_engines)} model-building engines.")

    # --- PGN and Session Handling ---
    pgn_path_str = input(f"Enter the full path to your PGN file: ")
    pgn_path = Path(pgn_path_str)
    if not pgn_path.is_file():
        logging.error(f"PGN file not found at {pgn_path}")
        return

    session_data = load_session(pgn_path)
    if not session_data:
        session_data = {
            'pgn_file': str(pgn_path),
            'games_to_process_indices': [],
            'completed_games_data': [],
            'in_progress_game': {}
        }
        with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
            while True:
                offset = pgn_file.tell()
                game = chess.pgn.read_game(pgn_file)
                if game is None: break
                white_player = game.headers.get("White", "?")
                black_player = game.headers.get("Black", "?")
                if PLAYER_TO_ANALYZE.lower() in (white_player.lower(), black_player.lower()):
                    session_data['games_to_process_indices'].append(offset)
        save_session(session_data, pgn_path)

    total_games_to_analyze = len(session_data['games_to_process_indices'])
    logging.info(f"Found {total_games_to_analyze} games remaining to analyze.")

    progress_logger.info(f"Chess Analysis Progress Report")
    progress_logger.info(f"Session started: {SESSION_TIMESTAMP}")
    progress_logger.info(f"Player to analyze: {PLAYER_TO_ANALYZE}")
    progress_logger.info(f"Found {total_games_to_analyze} games to analyze in '{pgn_path.name}'.")
    progress_logger.info("-" * 50)

    # --- Main Game Loop ---
    continuous_mode = False
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        while session_data['games_to_process_indices']:
            offset = session_data['games_to_process_indices'][0]
            completed_count = len(session_data['completed_games_data'])
            game_num_overall = completed_count + 1

            pgn_file.seek(offset)
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                 logging.error(f"Could not read game at offset {offset}. Exiting.")
                 session_data['games_to_process_indices'].pop(0)
                 save_session(session_data, pgn_path)
                 continue
            
            game.offset = offset

            if session_data.get('in_progress_game', {}).get('offset') != offset:
                session_data['in_progress_game'] = {'offset': offset}

            white = game.headers.get('White', '?')
            black = game.headers.get('Black', '?')
            logging.info(f"--- Starting Analysis for Game {game_num_overall}: {white} vs. {black} ---")
            progress_logger.info(f"\nGame {game_num_overall}/{total_games_to_analyze}: Analyzing {white} vs. {black}...")

            ground_truth_evals = get_averaged_ground_truth(game, oracle_engines, session_data, pgn_path)
            if not ground_truth_evals:
                logging.warning(f"Skipping Game {game_num_overall}: Could not generate ground truth.")
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data,pgn_path)
                continue

            model_results = build_model_with_real_engines(ground_truth_evals, model_engines, session_data, pgn_path)
            if model_results[0] is None:
                logging.warning(f"Skipping Game {game_num_overall} due to failure in model generation.")
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data,pgn_path)
                continue

            model, r_squared, ratings, quality_scores, model_type, raw_model_results = model_results

            plt.figure(figsize=(10, 6))
            plt.scatter(ratings.flatten(), quality_scores, alpha=0.7, label="Engine Performance (Move Quality)")
            
            plot_ratings_x = np.linspace(ratings.min(), ratings.max(), 200).reshape(-1, 1)
            plot_qualities_y = model.predict(plot_ratings_x)
            plt.plot(plot_ratings_x, plot_qualities_y, color='red', linewidth=2, label=f"{model_type} Regression Model (R²={r_squared:.4f})")
            
            plt.title(f"Game {game_num_overall}: Engine Rating vs. Move Quality")
            plt.xlabel("Engine Elo Rating (from CSV)")
            plt.ylabel("Average Move Quality")
            plt.grid(True); plt.legend()
            graph_path = SESSION_FOLDER / f"performance_graph_game_{game_num_overall}.png"
            plt.savefig(graph_path); plt.close()

            avg_quality, moves_counted = analyze_player_quality(game, ground_truth_evals, oracle_engines[0])

            if moves_counted == 0:
                logging.warning(f"Skipping rating estimation for Game {game_num_overall} as no moves were analyzed for the player.")
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data,pgn_path)
                continue

            est_rating = estimate_rating_from_quality(model, avg_quality, model_type)
            
            progress_logger.info(f"-> SUCCESS. Estimated Rating: {est_rating}. (Model: {model_type}, R²={r_squared:.3f} on {moves_counted} moves)")

            game_data_for_session = {
                'game_num': game_num_overall, 'white': white, 'black': black,
                'r_squared': r_squared, 'graph_path': str(graph_path),
                'model_type': model_type,
                'avg_quality': avg_quality, 'moves': moves_counted,
                'estimated_rating': est_rating,
                'model_results_raw': raw_model_results
            }
            session_data['completed_games_data'].append(game_data_for_session)
            session_data['games_to_process_indices'].pop(0)
            session_data['in_progress_game'] = {}
            save_session(session_data, pgn_path)
            logging.info(f"--- Finished Analysis for Game {game_num_overall}. Estimated rating: {est_rating}. ---")
            
            if not continuous_mode and session_data['games_to_process_indices']:
                user_input = input("Continue with next game? (y/n/c for yes/no/continuous): ").lower().strip()
                if user_input == 'n':
                    break
                if user_input == 'c':
                    continuous_mode = True

    logging.info("All games have been analyzed or queue is empty.")
    
    completed_games_data = session_data.get('completed_games_data', [])
    if completed_games_data:
        player_ratings = [g['estimated_rating'] for g in completed_games_data if 'estimated_rating' in g]
        if player_ratings:
            avg_rating = int(np.mean(player_ratings))
            summary_line = f"\nFinal Average Estimated Rating: {avg_rating} Elo across {len(player_ratings)} games."
            progress_logger.info("-" * 50)
            progress_logger.info(summary_line)
            logging.info(summary_line)

    generate_final_report(session_data)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
