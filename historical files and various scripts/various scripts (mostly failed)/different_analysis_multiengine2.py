# chess_analyzer_multi_engine_depth.py
# A script to analyze PGN chess games and estimate a specific player's rating.
# This version uses a sophisticated multi-engine, depth-based approach to
# achieve a high R-squared correlation.
#
# KEY FEATURES:
#   1. REAL ENGINE PANEL: Uses a CSV file of real engines with known ratings to
#      generate high-quality, non-random data for the performance model.
#   2. AVERAGED GROUND TRUTH: An "oracle panel" of the strongest engines provides
#      an averaged evaluation for each position, creating a stable baseline.
#   3. MOVE QUALITY METRIC: A nuanced (0.0-1.0) score based on centipawn
#      loss provides a richer signal than a simple hit/miss system.
#   4. DEPTH-BASED ANALYSIS: Uses search depth instead of time for consistent effort.
#   5. STABLE & INTERACTIVE: Uses a fault-tolerant, isolated process model and
#      is fully interactive and resumable.

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

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
ENGINES_CSV_PATH = PROJECT_FOLDER / "real_engines.csv"
# DEPTH-BASED ANALYSIS CONTROLS
ORACLE_ANALYSIS_DEPTH = 20
MODEL_BUILD_DEPTH = 12

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
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
            if isinstance(info, dict) and 'score' in info:
                score_obj = info['score'].white()
                score = score_obj.score(mate_score=30000)
                return (fen, engine_info['name'], score)
    except Exception as e:
        logging.error(f"Worker for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
    return (fen, engine_info['name'], None)

def get_move_worker(args):
    """Worker for a single play task. Returns the move played."""
    fen, engine_info, depth = args
    board = chess.Board(fen)
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            result = engine.play(board, chess.engine.Limit(depth=depth))
            if result.move:
                return (fen, engine_info['name'], result.move)
    except Exception as e:
        logging.error(f"Worker for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
    return (fen, engine_info['name'], None)

# --- MAIN ANALYSIS FUNCTIONS ---

def calculate_move_quality(centipawn_loss):
    """Converts centipawn loss to a quality score between 0.0 and 1.0."""
    # A simple linear scale for robustness
    if centipawn_loss <= 10: return 1.0
    if centipawn_loss >= 150: return 0.0
    return 1.0 - ((centipawn_loss - 10) / 140)

def get_averaged_ground_truth(game, oracle_engines, num_processes):
    """
    Gets an averaged evaluation for each position from the oracle panel.
    """
    logging.info(f"Generating averaged ground truth with {len(oracle_engines)} oracle engines...")
    
    fens_to_analyze = []
    board = game.board()
    player_color = chess.WHITE if game.headers.get("White") == PLAYER_TO_ANALYZE else chess.BLACK
    for move in game.mainline_moves():
        if board.turn == player_color:
            fens_to_analyze.append(board.fen())
        board.push(move)

    tasks = [(fen, engine_info, ORACLE_ANALYSIS_DEPTH) for fen in fens_to_analyze for engine_info in oracle_engines]
    
    position_evals = defaultdict(list)
    with multiprocessing.Pool(processes=num_processes) as pool:
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
    """
    logging.info(f"Building performance model with {len(model_engines)} real engines...")

    engine_qualities = defaultdict(list)
    with multiprocessing.Pool(processes=num_processes) as pool:
        for engine_info in model_engines:
            logging.info(f"  -> Testing model engine: {engine_info['name']}")
            tasks = [(fen, engine_info, MODEL_BUILD_DEPTH) for fen in ground_truth_evals.keys()]
            results = pool.map(get_move_worker, tasks)
            
            for fen, _, played_move in results:
                if played_move:
                    board = chess.Board(fen)
                    board.push(played_move)
                    # This is a simplification; a more accurate method would not re-launch the engine
                    with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as temp_engine:
                         info = temp_engine.analyse(board, chess.engine.Limit(depth=10))
                         if isinstance(info, dict) and 'score' in info:
                            played_move_score = info['score'].pov(-board.turn).score(mate_score=30000)
                            best_score_pov = ground_truth_evals[fen] if board.turn == chess.BLACK else -ground_truth_evals[fen]
                            cpl = max(0, best_score_pov - played_move_score)
                            engine_qualities[engine_info['name']].append(calculate_move_quality(cpl))

    ratings_data, quality_scores_data = [], []
    for engine_info in model_engines:
        name = engine_info['name']
        if engine_qualities[name]:
            avg_quality = np.mean(engine_qualities[name])
            ratings_data.append(engine_info['rating'])
            quality_scores_data.append(avg_quality)

    if len(ratings_data) < 2: return None, None, None, None

    ratings = np.array(ratings_data).reshape(-1, 1)
    quality_scores = np.array(quality_scores_data)
    model = LinearRegression()
    model.fit(ratings, quality_scores)
    r_squared = model.score(ratings, quality_scores)
    
    logging.info(f"Model created. R-squared: {r_squared:.4f}")
    return model, r_squared, ratings, quality_scores

# --- UTILITY AND REPORTING FUNCTIONS ---

def analyze_player_quality(game, ground_truth_evals):
    """Calculates the average move quality for the target player."""
    total_quality = 0
    moves_counted = 0
    
    board = game.board()
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        for move in game.mainline_moves():
            player_name = "White" if board.turn == chess.WHITE else "Black"
            if game.headers.get(player_name) == PLAYER_TO_ANALYZE:
                fen = board.fen()
                best_eval = ground_truth_evals.get(fen)
                if best_eval is not None:
                    board.push(move)
                    info = engine.analyse(board, chess.engine.Limit(depth=12))
                    if isinstance(info, dict) and 'score' in info:
                        played_move_score = info['score'].pov(-board.turn).score(mate_score=30000)
                        best_score_pov = best_eval if board.turn == chess.BLACK else -best_eval
                        cpl = max(0, best_score_pov - played_move_score)
                        total_quality += calculate_move_quality(cpl)
                        moves_counted += 1
                    board.pop()
                else:
                    board.push(move)
            else:
                board.push(move)

    avg_quality = (total_quality / moves_counted) if moves_counted > 0 else 0
    return avg_quality, moves_counted

def estimate_rating_from_quality(model, quality_score):
    """Estimates Elo rating from a quality score using the linear model."""
    m, c = model.coef_[0], model.intercept_
    if m <= 0: return 0
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
        with open(session_file, 'r') as f:
            return json.load(f)
    return None

def generate_final_report(session_data, session_folder):
    logging.info("Generating final PDF report...")
    all_games_data = session_data.get('completed_games_data', [])
    if not all_games_data:
        logging.warning("No completed game data to report.")
        return

    for data in all_games_data:
        if 'model_coef' in data and 'model_intercept' in data:
            model = LinearRegression()
            model.coef_ = np.array([data['model_coef']])
            model.intercept_ = data['model_intercept']
            data['model'] = model

    player_ratings = [g['estimated_rating'] for g in all_games_data if 'estimated_rating' in g]
    final_average_data = None
    if player_ratings:
        avg_rating = int(np.mean(player_ratings))
        final_average_data = {
            "player_name": PLAYER_TO_ANALYZE,
            "game_count": len(player_ratings),
            "average_rating": avg_rating
        }

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(w=0, h=10, text="Chess Performance Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)

        for i, data in enumerate(all_games_data):
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(w=0, h=10, text=f"Analysis for Game {data['game_num']}: {data['white']} vs. {data['black']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            if Path(data['graph_path']).is_file():
                pdf.image(data['graph_path'], w=180)
            
            pdf.ln(5)
            pdf.set_font("Helvetica", '', 10)
            if 'model' in data:
                m, c = data['model'].coef_[0], data['model'].intercept_
                pdf.multi_cell(w=0, h=5, text=f"Model Equation: Move Quality = {m:.6f} * Rating + {c:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.multi_cell(w=0, h=5, text=f"Model Fit (R-squared): {data['r_squared']:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            
            if 'avg_quality' in data:
                pdf.set_font("Helvetica", 'B', 11)
                pdf.cell(w=0, h=8, text=f"Player: {PLAYER_TO_ANALYZE}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", '', 10)
                pdf.multi_cell(w=0, h=5, text=f"  - Average Move Quality: {data['avg_quality']:.2%}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.multi_cell(w=0, h=5, text=f"  - Estimated Game Rating: {data['estimated_rating']} Elo", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.ln(5)
            if (i + 1) < len(all_games_data):
                 pdf.add_page()

        if final_average_data:
            if len(all_games_data) > 0: pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(w=0, h=10, text="Overall Performance Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            pdf.ln(10)
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(w=0, h=10, text=f"Player: {final_average_data['player_name']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", '', 10)
            pdf.multi_cell(w=0, h=5, text=f"Analyzed across {final_average_data['game_count']} games.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.multi_cell(w=0, h=5, text=f"Average Estimated Performance Rating: {final_average_data['average_rating']} Elo", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf_path = session_folder / f"Chess_Analysis_Report_FINAL.pdf"
        pdf.output(pdf_path)
        logging.info(f"Final report saved to {pdf_path}")
    except Exception as e:
        logging.error(f"Failed to generate final PDF report. Error: {repr(e)}")


# --- MAIN EXECUTION LOGIC ---

def main():
    # --- Load Engines from CSV ---
    all_engines = []
    with open(ENGINES_CSV_PATH, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            row['rating'] = int(row['rating'])
            all_engines.append(row)
    
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
                game = chess.pgn.read_game(pgn_file)
                if game is None: break
                if PLAYER_TO_ANALYZE in (game.headers.get("White"), game.headers.get("Black")):
                    session_data['games_to_process_indices'].append(offset)
        save_session(session_data, pgn_path, SESSION_FOLDER)

    games_to_process_offsets = session_data['games_to_process_indices']
    logging.info(f"Found {len(games_to_process_offsets)} games remaining to analyze.")

    # --- Main Game Loop ---
    continuous_mode = False
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        for i, offset in enumerate(list(games_to_process_offsets)):
            completed_count = len(session_data['completed_games_data'])
            game_num = completed_count + 1
            pgn_file.seek(offset)
            game = chess.pgn.read_game(pgn_file)
            
            logging.info(f"--- Starting Analysis for Game {game_num}: {game.headers.get('White', '?')} vs. {game.headers.get('Black', '?')} ---")

            # Step 1: Generate the averaged ground truth for this game
            ground_truth_evals = get_averaged_ground_truth(game, oracle_engines, len(oracle_engines))
            
            # Step 2: Build the performance model using the model engines
            model_results = build_model_with_real_engines(ground_truth_evals, model_engines, len(model_engines))
            if model_results[0] is None:
                logging.warning(f"Skipping Game {game_num} due to failure in model generation.")
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data, pgn_path, SESSION_FOLDER)
                continue
            
            model, r_squared, ratings, quality_scores = model_results
            
            # Step 3: Plotting
            plt.figure(figsize=(10, 6))
            plt.scatter(ratings, quality_scores, alpha=0.7, label="Engine Performance (Move Quality)")
            plt.plot(ratings, model.predict(ratings), color='red', linewidth=2, label="Linear Regression Model")
            plt.title(f"Game {game_num}: Engine Rating vs. Move Quality")
            plt.xlabel("Engine Elo Rating (from CSV)")
            plt.ylabel("Average Move Quality")
            plt.grid(True); plt.legend()
            graph_path = SESSION_FOLDER / f"performance_graph_game_{game_num}.png"
            plt.savefig(graph_path); plt.close()

            # Step 4: Analyze human player
            avg_quality, moves_counted = analyze_player_quality(game, ground_truth_evals)
            est_rating = estimate_rating_from_quality(model, avg_quality)
            
            # Step 5: Save results
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
            logging.info(f"--- Finished Analysis for Game {game_num}. Results saved. ---")

            # Step 6: Interactive Prompt
            if not continuous_mode:
                while True:
                    user_input = input("Continue analysis? (y/n), Mode? (c/b for continuous/batch): ").lower().strip()
                    parts = [p.strip() for p in user_input.split(',')]
                    if len(parts) == 2 and parts[0] in ['y', 'n'] and parts[1] in ['c', 'b']:
                        if parts[0] == 'n':
                            logging.info("Analysis paused by user.")
                            generate_final_report(session_data, SESSION_FOLDER)
                            return
                        if parts[1] == 'c':
                            continuous_mode = True
                            logging.info("Switching to continuous mode.")
                        break
                    else:
                        print("Invalid input. Please use format: y,c or n,b etc.")
    
    logging.info("All games have been analyzed.")
    generate_final_report(session_data, SESSION_FOLDER)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
