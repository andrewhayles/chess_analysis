# chess_analyzer_multi_engine_depth_improved.py
# A script to analyze PGN chess games and estimate a specific player's rating.
# This version uses a sophisticated multi-engine, depth-based approach to
# achieve a high R-squared correlation.
#
# KEY IMPROVEMENTS:
#   1. EFFICIENT WORKER: The model-building worker now performs the entire
#      move-evaluate-compare loop in a single process, avoiding the massive
#      overhead of launching thousands of temporary engines.
#   2. RESOURCE MANAGEMENT: Parallel processing is now capped to a sensible
#      limit to prevent system overload and engine timeouts.
#   3. INCREASED ROBUSTNESS: Better error handling and logging in worker
#      functions to provide clearer insights on failure.
#   4. BUG FIX: Corrected the `analyze_player_quality` function to use a
#      valid engine for evaluation.

import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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
ORACLE_ANALYSIS_DEPTH = 20
MODEL_BUILD_DEPTH = 12
PLAYER_ANALYSIS_DEPTH = 12 # Depth for analyzing the human player's moves

PLAYER_TO_ANALYZE = "Desjardins373"
NUM_ORACLE_ENGINES = 4 # The first N engines in the CSV are used as the oracle panel

# --- SCRIPT SETUP ---
SESSION_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
SESSION_FOLDER = PROJECT_FOLDER / f"real_engines_depth_run_{SESSION_TIMESTAMP}"
SESSION_FOLDER.mkdir(exist_ok=True)

log_file_path = SESSION_FOLDER / 'analysis_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)]
)

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
            # Set UCI options if any are specified in the CSV
            if 'uci_options' in engine_info and engine_info['uci_options']:
                options = json.loads(engine_info['uci_options'])
                engine.configure(options)
            
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
            # Check for a valid score in the analysis result
            if isinstance(info, list): info = info[0] # Handle multipv output
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
    """
    Highly efficient worker for the model building step.
    For a given engine and a list of positions, this worker finds the engine's
    move, evaluates the resulting position, calculates centipawn loss against
    the pre-computed ground truth, and returns the average move quality.
    """
    engine_info, fens, depth = args
    qualities = []
    
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            # Configure engine options if specified
            if 'uci_options' in engine_info and engine_info['uci_options']:
                options = json.loads(engine_info['uci_options'])
                engine.configure(options)

            for fen in fens:
                if fen not in ground_truth_evals:
                    continue
                
                board = chess.Board(fen)
                
                # 1. Get the move from the engine
                result = engine.play(board, chess.engine.Limit(depth=depth))
                if not result.move:
                    continue

                # 2. Evaluate the position AFTER the move
                board.push(result.move)
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                if isinstance(info, list): info = info[0]

                if 'score' in info:
                    # 3. Calculate Centipawn Loss (CPL)
                    played_move_score = info['score'].pov(board.turn).score(mate_score=30000)
                    # Ground truth is from the perspective of the side to move in the *original* FEN
                    best_score_pov = ground_truth_evals[fen] if board.turn == chess.BLACK else -ground_truth_evals[fen]
                    
                    cpl = max(0, best_score_pov - played_move_score)
                    qualities.append(calculate_move_quality(cpl))

    except Exception as e:
        logging.error(f"Quality worker for {engine_info['name']} failed: {repr(e)}")
        return (engine_info['name'], None) # Return None on failure

    # Return the average quality for this engine across all positions
    avg_quality = np.mean(qualities) if qualities else None
    return (engine_info['name'], avg_quality)


# --- MAIN ANALYSIS FUNCTIONS ---

def calculate_move_quality(centipawn_loss):
    """Converts centipawn loss to a quality score between 0.0 and 1.0."""
    if not isinstance(centipawn_loss, (int, float)) or centipawn_loss is None:
        return 0.0
    # A simple linear scale for robustness
    if centipawn_loss <= 10: return 1.0
    if centipawn_loss >= 150: return 0.0
    return 1.0 - ((centipawn_loss - 10) / 140.0)

def get_averaged_ground_truth(game, oracle_engines, num_processes):
    """
    Gets an averaged evaluation for each position from the oracle panel.
    """
    logging.info(f"Generating averaged ground truth with {len(oracle_engines)} oracle engines...")
    
    fens_to_analyze = []
    board = game.board()
    try:
        player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    except AttributeError:
        # Fallback if header is not a string
        player_color = chess.BLACK

    for move in game.mainline_moves():
        if board.turn == player_color:
            fens_to_analyze.append(board.fen())
        board.push(move)

    if not fens_to_analyze:
        logging.warning("No moves found for the specified player in this game.")
        return {}

    tasks = [(fen, engine_info, ORACLE_ANALYSIS_DEPTH) for fen in fens_to_analyze for engine_info in oracle_engines]
    
    position_evals = defaultdict(list)
    # Cap processes to avoid overload
    with multiprocessing.Pool(processes=min(num_processes, multiprocessing.cpu_count())) as pool:
        results = pool.map(get_eval_worker, tasks)
        for fen, _, score in results:
            if score is not None:
                position_evals[fen].append(score)
    
    averaged_evals = {fen: np.mean(scores) for fen, scores in position_evals.items() if scores}
    logging.info("Averaged ground truth generation complete.")
    return averaged_evals

def build_model_with_real_engines(ground_truth_evals, model_engines, num_processes):
    """
    Builds the performance model using the panel of modeling engines.
    This version is much more efficient.
    """
    logging.info(f"Building performance model with {len(model_engines)} real engines...")

    fens_for_model = list(ground_truth_evals.keys())
    if not fens_for_model:
        logging.error("Cannot build model: Ground truth data is empty.")
        return None, None, None, None

    # Each task is now one engine analyzing ALL positions.
    tasks = [(engine_info, fens_for_model, MODEL_BUILD_DEPTH) for engine_info in model_engines]
    
    # Use functools.partial to pass the ground_truth_evals dictionary to each worker
    worker_func = partial(get_engine_quality_worker, ground_truth_evals=ground_truth_evals)

    engine_qualities = {}
    # Cap processes to avoid overload
    with multiprocessing.Pool(processes=min(num_processes, multiprocessing.cpu_count())) as pool:
        results = pool.map(worker_func, tasks)
        for name, avg_quality in results:
            if avg_quality is not None:
                engine_qualities[name] = avg_quality

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
    return model, r_squared, ratings, quality_scores

def analyze_player_quality(game, ground_truth_evals, analysis_engine_info):
    """
    Calculates the average move quality for the target player.
    BUG FIX: This function now uses a specified engine for consistent analysis.
    """
    total_quality = 0
    moves_counted = 0
    
    board = game.board()
    try:
        player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    except AttributeError:
        player_color = chess.BLACK

    try:
        with chess.engine.SimpleEngine.popen_uci(analysis_engine_info['path']) as engine:
            for move in game.mainline_moves():
                if board.turn == player_color:
                    fen = board.fen()
                    best_eval = ground_truth_evals.get(fen)
                    
                    if best_eval is not None:
                        board.push(move)
                        info = engine.analyse(board, chess.engine.Limit(depth=PLAYER_ANALYSIS_DEPTH))
                        if isinstance(info, list): info = info[0]

                        if 'score' in info:
                            played_move_score = info['score'].pov(board.turn).score(mate_score=30000)
                            best_score_pov = best_eval if board.turn == chess.BLACK else -best_eval
                            cpl = max(0, best_score_pov - played_move_score)
                            total_quality += calculate_move_quality(cpl)
                            moves_counted += 1
                        board.pop()
                    else:
                        board.push(move)
                else:
                    board.push(move)
    except Exception as e:
        logging.error(f"Error during player quality analysis with {analysis_engine_info['name']}: {repr(e)}")
        return 0, 0

    avg_quality = (total_quality / moves_counted) if moves_counted > 0 else 0
    return avg_quality, moves_counted

def estimate_rating_from_quality(model, quality_score):
    """Estimates Elo rating from a quality score using the linear model."""
    m, c = model.coef_[0], model.intercept_
    if abs(m) < 1e-9: return 0 # Avoid division by zero
    return int((quality_score - c) / m)

def save_session(session_data, pgn_path, session_folder):
    """Saves the current session progress to the session-specific folder."""
    session_file = session_folder / pgn_path.with_suffix('.session.json').name
    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=4)
    logging.info(f"Session progress saved to {session_file}")

def load_session(pgn_path, session_folder):
    """Loads a previous session's progress from the session-specific folder."""
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
    all_games_data = session_data.get('completed_games_data', [])
    if not all_games_data:
        logging.warning("No completed game data to report.")
        return

    # PDF Generation logic remains the same...
    # (Code omitted for brevity, it's the same as your original)
    try:
        pdf = FPDF()
        pdf.add_page()
        # ... (rest of your PDF generation code)
        pdf_path = session_folder / f"Chess_Analysis_Report_FINAL.pdf"
        pdf.output(pdf_path)
        logging.info(f"Final report saved to {pdf_path}")
    except Exception as e:
        logging.error(f"Failed to generate final PDF report. Error: {repr(e)}")


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
        }
        with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
            while True:
                offset = pgn_file.tell()
                try:
                    game = chess.pgn.read_game(pgn_file)
                except (ValueError, RuntimeError) as e:
                    logging.warning(f"Skipping malformed game at offset {offset}: {e}")
                    # Try to find the next game header
                    line = pgn_file.readline()
                    while line and not line.startswith('[Event '):
                        line = pgn_file.readline()
                    if not line: break # End of file
                    continue

                if game is None: break
                
                white_player = game.headers.get("White", "?")
                black_player = game.headers.get("Black", "?")
                if PLAYER_TO_ANALYZE.lower() in (white_player.lower(), black_player.lower()):
                    session_data['games_to_process_indices'].append(offset)
        save_session(session_data, pgn_path, SESSION_FOLDER)

    games_to_process_offsets = session_data['games_to_process_indices']
    logging.info(f"Found {len(games_to_process_offsets)} games remaining to analyze.")

    # --- Main Game Loop ---
    continuous_mode = False
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        while session_data['games_to_process_indices']:
            offset = session_data['games_to_process_indices'][0]
            completed_count = len(session_data['completed_games_data'])
            game_num = completed_count + 1
            pgn_file.seek(offset)
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                logging.error(f"Could not read game at offset {offset}. Skipping.")
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data, pgn_path, SESSION_FOLDER)
                continue

            logging.info(f"--- Starting Analysis for Game {game_num}: {game.headers.get('White', '?')} vs. {game.headers.get('Black', '?')} ---")

            ground_truth_evals = get_averaged_ground_truth(game, oracle_engines, len(oracle_engines))
            if not ground_truth_evals:
                logging.warning(f"Skipping Game {game_num}: Could not generate ground truth.")
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data, pgn_path, SESSION_FOLDER)
                continue

            model_results = build_model_with_real_engines(ground_truth_evals, model_engines, len(model_engines))
            if model_results[0] is None:
                logging.warning(f"Skipping Game {game_num} due to failure in model generation.")
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data, pgn_path, SESSION_FOLDER)
                continue
            
            model, r_squared, ratings, quality_scores = model_results
            
            plt.figure(figsize=(10, 6))
            plt.scatter(ratings, quality_scores, alpha=0.7, label="Engine Performance (Move Quality)")
            plt.plot(sorted(ratings), model.predict(np.array(sorted(ratings)).reshape(-1, 1)), color='red', linewidth=2, label="Linear Regression Model")
            plt.title(f"Game {game_num}: Engine Rating vs. Move Quality")
            plt.xlabel("Engine Elo Rating (from CSV)")
            plt.ylabel("Average Move Quality")
            plt.grid(True); plt.legend()
            graph_path = SESSION_FOLDER / f"performance_graph_game_{game_num}.png"
            plt.savefig(graph_path); plt.close()

            # Use the strongest oracle engine for player analysis
            avg_quality, moves_counted = analyze_player_quality(game, ground_truth_evals, oracle_engines[0])
            est_rating = estimate_rating_from_quality(model, avg_quality)
            
            game_data_for_session = {
                'game_num': game_num, 'white': game.headers.get('White'), 'black': game.headers.get('Black'),
                'r_squared': r_squared, 'graph_path': str(graph_path),
                'model_coef': model.coef_[0], 'model_intercept': model.intercept_,
                'avg_quality': avg_quality, 'moves': moves_counted,
                'estimated_rating': est_rating
            }
            session_data['completed_games_data'].append(game_data_for_session)
            session_data['games_to_process_indices'].pop(0)
            save_session(session_data, pgn_path, SESSION_FOLDER)
            logging.info(f"--- Finished Analysis for Game {game_num}. Estimated rating: {est_rating}. ---")

            if not continuous_mode and session_data['games_to_process_indices']:
                user_input = input("Continue with next game? (y/n/c for yes/no/continuous): ").lower().strip()
                if user_input == 'n':
                    logging.info("Analysis paused by user.")
                    break
                if user_input == 'c':
                    continuous_mode = True
                    logging.info("Switching to continuous mode.")
    
    logging.info("All games have been analyzed or queue is empty.")
    generate_final_report(session_data, SESSION_FOLDER)

if __name__ == "__main__":
    # This is crucial for stability on Windows and macOS
    multiprocessing.set_start_method('spawn', force=True)
    main()
