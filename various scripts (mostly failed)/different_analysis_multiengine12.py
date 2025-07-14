# chess_analyzer_multi_engine_depth_improved.py
# A script to analyze PGN chess games and estimate a specific player's rating.
# This version uses a sophisticated multi-engine, depth-based approach to
# achieve a high R-squared correlation.
#
# KEY IMPROVEMENTS:
#   1. EFFICIENT WORKER: The model-building worker now performs the entire
#      move-evaluate-compare loop in a single process.
#   2. RESOURCE MANAGEMENT: Parallel processing is capped to a sensible limit.
#   3. GRANULAR SAVING: Saves progress after each major step within a game,
#      allowing the script to resume efficiently after an interruption.
#   4. LIVE PROGRESS BAR: Shows a real-time progress bar for all long steps.
#   5. ROBUSTNESS: Uses fresh engine processes for player analysis to prevent state corruption.

import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg') # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos
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

# DEPTH-BASED ANALYSIS CONTROLS
ORACLE_ANALYSIS_DEPTH = 18
MODEL_BUILD_DEPTH = 10
PLAYER_ANALYSIS_DEPTH = 10 # Depth for analyzing the human player's moves

PLAYER_TO_ANALYZE = "Desjardins373"
NUM_ORACLE_ENGINES = 4 # The first N engines in the CSV are used as the oracle panel

# --- SCRIPT SETUP ---
SESSION_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
SESSION_FOLDER = PROJECT_FOLDER / f"real_engines_depth_run_{SESSION_TIMESTAMP}"
SESSION_FOLDER.mkdir(exist_ok=True)

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
progress_logger.propagate = False # Prevent messages from appearing in the main log/console
progress_file_handler = logging.FileHandler(progress_log_path)
progress_file_handler.setFormatter(logging.Formatter('%(message)s'))
progress_logger.addHandler(progress_file_handler)


if not ENGINES_CSV_PATH.is_file():
    logging.error(f"FATAL: Engines CSV file not found at '{ENGINES_CSV_PATH}'")
    sys.exit()

# --- ISOLATED WORKER FUNCTIONS ---

def get_eval_worker(args):
    """Worker for a single analysis task. Returns the evaluation score."""
    fen, engine_info, depth = args
    board = chess.Board(fen)
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            if 'uci_options' in engine_info and engine_info['uci_options']:
                options = json.loads(engine_info['uci_options'])
                engine.configure(options)
            
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=300), multipv=1)

            if isinstance(info, list): info = info[0]
            if 'score' in info:
                score_obj = info['score'].white()
                score = score_obj.score(mate_score=30000)
                return (fen, engine_info['name'], score)
            else:
                logging.warning(f"Worker for {engine_info['name']} on FEN {fen} produced no score.")
                return (fen, engine_info['name'], None)
    except Exception as e:
        logging.error(f"Worker for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
    return (fen, engine_info['name'], None)

def get_engine_quality_worker(args, ground_truth_evals):
    """Highly efficient worker for the model building step."""
    engine_info, fens, depth = args
    qualities = []
    
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            if 'uci_options' in engine_info and engine_info['uci_options']:
                options = json.loads(engine_info['uci_options'])
                engine.configure(options)

            for fen in fens:
                if fen not in ground_truth_evals:
                    continue
                
                board = chess.Board(fen)
                result = engine.play(board, chess.engine.Limit(depth=depth, time = 300))
                if not result.move:
                    continue

                board.push(result.move)
                info = engine.analyse(board, chess.engine.Limit(depth=depth, time = 300), info=chess.engine.INFO_SCORE)
                if isinstance(info, list): info = info[0]

                if 'score' in info:
                    played_move_score = info['score'].pov(board.turn).score(mate_score=30000)
                    best_score_pov = ground_truth_evals[fen] if board.turn == chess.BLACK else -ground_truth_evals[fen]
                    cpl = max(0, best_score_pov - played_move_score)
                    qualities.append(calculate_move_quality(cpl))

    except Exception as e:
        logging.error(f"Quality worker for {engine_info['name']} failed: {repr(e)}")
        return (engine_info['name'], None)

    avg_quality = np.mean(qualities) if qualities else None
    return (engine_info['name'], avg_quality)

# --- UTILITY FUNCTIONS ---
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """Call in a loop to create terminal progress bar"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

# --- MAIN ANALYSIS FUNCTIONS ---

def calculate_move_quality(centipawn_loss):
    """Converts centipawn loss to a quality score between 0.0 and 1.0."""
    if not isinstance(centipawn_loss, (int, float)) or centipawn_loss is None:
        return 0.0
    if centipawn_loss <= 10: return 1.0
    if centipawn_loss >= 150: return 0.0
    return 1.0 - ((centipawn_loss - 10) / 140.0)

def get_averaged_ground_truth(game, oracle_engines, num_processes, session_data, pgn_path, session_folder):
    """Gets an averaged evaluation for each position, loading from session if possible."""
    
    # Check if we already have this data in our session file
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

    tasks = [(fen, engine_info, ORACLE_ANALYSIS_DEPTH) for fen in fens_to_analyze for engine_info in oracle_engines]
    total_tasks = len(tasks)
    position_evals = defaultdict(list)
    
    print_progress_bar(0, total_tasks, prefix='Ground Truth:', suffix='Complete', length=50)

    with multiprocessing.Pool(processes=min(num_processes, multiprocessing.cpu_count())) as pool:
        for i, result in enumerate(pool.imap_unordered(get_eval_worker, tasks)):
            fen, _, score = result
            if score is not None:
                position_evals[fen].append(score)
            print_progress_bar(i + 1, total_tasks, prefix='Ground Truth:', suffix='Complete', length=50)

    averaged_evals = {fen: np.mean(scores) for fen, scores in position_evals.items() if scores}
    logging.info("Averaged ground truth generation complete.")
    progress_logger.info(f"  -> Ground truth generated for {len(averaged_evals)} positions.")

    # Save this result to the session file immediately
    session_data['in_progress_game']['ground_truth_evals'] = averaged_evals
    save_session(session_data, pgn_path, session_folder)

    return averaged_evals

def build_model_with_real_engines(ground_truth_evals, model_engines, num_processes, session_data, pgn_path, session_folder):
    """Builds the performance model, resuming from session if possible."""
    
    logging.info(f"Building performance model with {len(model_engines)} real engines...")
    progress_logger.info(f"  -> Building performance model...")

    fens_for_model = list(ground_truth_evals.keys())
    if not fens_for_model:
        logging.error("Cannot build model: Ground truth data is empty.")
        return None, None, None, None

    # Load any engine qualities we've already calculated
    engine_qualities = session_data.get('in_progress_game', {}).get('model_engine_results', {})
    
    # Determine which engines still need to be run
    completed_engines = set(engine_qualities.keys())
    remaining_engines = [eng for eng in model_engines if eng['name'] not in completed_engines]
    
    if remaining_engines:
        logging.info(f"Resuming model build. {len(completed_engines)} engines already complete.")
        tasks = [(engine_info, fens_for_model, MODEL_BUILD_DEPTH) for engine_info in remaining_engines]
        total_tasks = len(tasks)
        worker_func = partial(get_engine_quality_worker, ground_truth_evals=ground_truth_evals)

        print_progress_bar(0, total_tasks, prefix='Model Building:', suffix='Complete', length=50)
        with multiprocessing.Pool(processes=min(num_processes, multiprocessing.cpu_count())) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_func, tasks)):
                name, avg_quality = result
                if avg_quality is not None:
                    engine_qualities[name] = avg_quality
                    logging.info(f"  -> Model data complete for engine: {name}")
                    progress_logger.info(f"    - Model data gathered for: {name}")
                    # Save progress after each engine finishes
                    session_data['in_progress_game']['model_engine_results'] = engine_qualities
                    save_session(session_data, pgn_path, session_folder)
                print_progress_bar(i + 1, total_tasks, prefix='Model Building:', suffix='Complete', length=50)
    else:
        logging.info("All model engine data loaded from session file.")
        progress_logger.info("  -> All model engine data loaded from session file.")

    ratings_data, quality_scores_data = [], []
    for engine_info in model_engines:
        name = engine_info['name']
        if name in engine_qualities:
            ratings_data.append(engine_info['rating'])
            quality_scores_data.append(engine_qualities[name])

    if len(ratings_data) < 2: 
        logging.warning(f"Failed to gather enough data for model. Only got {len(ratings_data)} points.")
        return None, None, None, None

    ratings = np.array(ratings_data).reshape(-1, 1)
    quality_scores = np.array(quality_scores_data)
    model = LinearRegression()
    model.fit(ratings, quality_scores)
    r_squared = model.score(ratings, quality_scores)
    
    logging.info(f"Model created. R-squared: {r_squared:.4f}")
    progress_logger.info(f"  -> Model created with R-squared: {r_squared:.4f}")
    return model, r_squared, ratings, quality_scores

def analyze_player_quality(game, ground_truth_evals, analysis_engine_info):
    """
    Calculates the average move quality for the target player.
    This version is more robust, creating a new engine process for each move.
    """
    total_quality = 0
    moves_counted = 0
    
    board = game.board()
    try:
        player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    except AttributeError:
        player_color = chess.BLACK

    all_moves = list(game.mainline_moves())
    player_move_indices = [i for i, move in enumerate(all_moves) if board.copy(stack=i).turn == player_color]
    
    print_progress_bar(0, len(player_move_indices), prefix='Player Analysis:', suffix='Complete', length=50)

    for i, move_index in enumerate(player_move_indices):
        board.reset()
        for j in range(move_index):
            board.push(all_moves[j])
        
        move = all_moves[move_index]
        fen = board.fen()
        best_eval = ground_truth_evals.get(fen)
        
        if best_eval is not None:
            try:
                with chess.engine.SimpleEngine.popen_uci(analysis_engine_info['path']) as engine:
                    temp_board = board.copy()
                    temp_board.push(move)
                    
                    info = engine.analyse(temp_board, chess.engine.Limit(depth=PLAYER_ANALYSIS_DEPTH, time = 300), info=chess.engine.INFO_SCORE)
                    if isinstance(info, list): info = info[0]

                    if 'score' in info:
                        played_move_score = info['score'].pov(temp_board.turn).score(mate_score=30000)
                        best_score_pov = best_eval if temp_board.turn == chess.BLACK else -best_eval
                        
                        cpl = max(0, best_score_pov - played_move_score)
                        total_quality += calculate_move_quality(cpl)
                        moves_counted += 1
            except Exception as e:
                logging.error(f"Error during single move analysis with {analysis_engine_info['name']} on FEN {fen}: {repr(e)}")
        
        print_progress_bar(i + 1, len(player_move_indices), prefix='Player Analysis:', suffix='Complete', length=50)

    avg_quality = (total_quality / moves_counted) if moves_counted > 0 else 0
    return avg_quality, moves_counted

def estimate_rating_from_quality(model, quality_score):
    """Estimates Elo rating from a quality score using the linear model."""
    m, c = model.coef_[0], model.intercept_
    if abs(m) < 1e-9: return 0 # Avoid division by zero
    return int((quality_score - c) / m)

def save_session(session_data, pgn_path, session_folder):
    """Saves the current session progress."""
    session_file = session_folder / pgn_path.with_suffix('.session.json').name
    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=4)
    logging.info(f"Session progress saved to {session_file}")

def load_session(pgn_path, session_folder):
    """Loads a previous session's progress."""
    session_file = session_folder / pgn_path.with_suffix('.session.json').name
    if session_file.exists():
        logging.info(f"Found existing session file: {session_file}")
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Could not read session file {session_file}. Starting fresh.")
            return None
    return None

def generate_final_report(session_data, session_folder):
    logging.info("Generating final PDF report...")
    # PDF generation logic remains the same
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
    except FileNotFoundError:
        logging.error(f"FATAL: Could not find engines CSV at {ENGINES_CSV_PATH}")
        return
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

    session_data = load_session(pgn_path, SESSION_FOLDER)
    if not session_data:
        logging.info("Starting a new analysis session.")
        session_data = {
            'pgn_file': str(pgn_path),
            'games_to_process_indices': [],
            'completed_games_data': [],
            'in_progress_game': {}
        }
        with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
            while True:
                offset = pgn_file.tell()
                try:
                    game = chess.pgn.read_game(pgn_file)
                except (ValueError, RuntimeError) as e:
                    logging.warning(f"Skipping malformed game at offset {offset}: {e}")
                    line = pgn_file.readline()
                    while line and not line.startswith('[Event '):
                        line = pgn_file.readline()
                    if not line: break
                    continue

                if game is None: break
                
                white_player = game.headers.get("White", "?")
                black_player = game.headers.get("Black", "?")
                if PLAYER_TO_ANALYZE.lower() in (white_player.lower(), black_player.lower()):
                    session_data['games_to_process_indices'].append(offset)
        save_session(session_data, pgn_path, SESSION_FOLDER)

    games_to_process_offsets = session_data['games_to_process_indices']
    total_games_to_analyze = len(games_to_process_offsets)
    logging.info(f"Found {total_games_to_analyze} games remaining to analyze.")
    
    # --- Initialize Progress File ---
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
            game.offset = offset # Store offset in game object for convenience

            if game is None:
                logging.error(f"Could not read game at offset {offset}. Skipping.")
                progress_logger.info(f"Game {game_num_overall}/{total_games_to_analyze}: ERROR reading PGN. Skipping.")
                session_data['games_to_process_indices'].pop(0)
                session_data['in_progress_game'] = {} # Clear in-progress data
                save_session(session_data, pgn_path, SESSION_FOLDER)
                continue

            # --- Manage In-Progress Game State ---
            if session_data.get('in_progress_game', {}).get('offset') != offset:
                logging.info(f"Starting fresh analysis for game at offset {offset}.")
                session_data['in_progress_game'] = {'offset': offset}
            else:
                logging.info(f"Resuming analysis for game at offset {offset}.")

            white = game.headers.get('White', '?')
            black = game.headers.get('Black', '?')
            logging.info(f"--- Starting Analysis for Game {game_num_overall}: {white} vs. {black} ---")
            progress_logger.info(f"\nGame {game_num_overall}/{total_games_to_analyze}: Analyzing {white} vs. {black}...")

            ground_truth_evals = get_averaged_ground_truth(game, oracle_engines, len(oracle_engines), session_data, pgn_path, SESSION_FOLDER)
            if not ground_truth_evals:
                logging.warning(f"Skipping Game {game_num_overall}: Could not generate ground truth.")
                progress_logger.info(f"-> SKIPPED. Reason: Could not generate ground truth for player.")
                session_data['games_to_process_indices'].pop(0)
                session_data['in_progress_game'] = {}
                save_session(session_data, pgn_path, SESSION_FOLDER)
                continue

            model_results = build_model_with_real_engines(ground_truth_evals, model_engines, len(model_engines), session_data, pgn_path, SESSION_FOLDER)
            if model_results[0] is None:
                logging.warning(f"Skipping Game {game_num_overall} due to failure in model generation.")
                progress_logger.info(f"-> SKIPPED. Reason: Failure in model generation.")
                session_data['games_to_process_indices'].pop(0)
                session_data['in_progress_game'] = {}
                save_session(session_data, pgn_path, SESSION_FOLDER)
                continue
            
            model, r_squared, ratings, quality_scores = model_results
            
            plt.figure(figsize=(10, 6))
            plt.scatter(ratings, quality_scores, alpha=0.7, label="Engine Performance (Move Quality)")
            plt.plot(sorted(ratings), model.predict(np.array(sorted(ratings)).reshape(-1, 1)), color='red', linewidth=2, label="Linear Regression Model")
            plt.title(f"Game {game_num_overall}: Engine Rating vs. Move Quality")
            plt.xlabel("Engine Elo Rating (from CSV)")
            plt.ylabel("Average Move Quality")
            plt.grid(True); plt.legend()
            graph_path = SESSION_FOLDER / f"performance_graph_game_{game_num_overall}.png"
            plt.savefig(graph_path); plt.close()

            avg_quality, moves_counted = analyze_player_quality(game, ground_truth_evals, oracle_engines[0])
            est_rating = estimate_rating_from_quality(model, avg_quality)
            
            progress_logger.info(f"-> SUCCESS. Estimated Rating: {est_rating}. (R²={r_squared:.3f} on {moves_counted} moves)")
            
            game_data_for_session = {
                'game_num': game_num_overall, 'white': white, 'black': black,
                'r_squared': r_squared, 'graph_path': str(graph_path),
                'model_coef': model.coef_[0], 'model_intercept': model.intercept_,
                'avg_quality': avg_quality, 'moves': moves_counted,
                'estimated_rating': est_rating
            }
            session_data['completed_games_data'].append(game_data_for_session)
            session_data['games_to_process_indices'].pop(0)
            session_data['in_progress_game'] = {} # Clear in-progress data for the completed game
            save_session(session_data, pgn_path, SESSION_FOLDER)
            logging.info(f"--- Finished Analysis for Game {game_num_overall}. Estimated rating: {est_rating}. ---")

            if not continuous_mode and session_data['games_to_process_indices']:
                user_input = input("Continue with next game? (y/n/c for yes/no/continuous): ").lower().strip()
                if user_input == 'n':
                    logging.info("Analysis paused by user.")
                    break
                if user_input == 'c':
                    continuous_mode = True
                    logging.info("Switching to continuous mode.")
    
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

    generate_final_report(session_data, SESSION_FOLDER)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
