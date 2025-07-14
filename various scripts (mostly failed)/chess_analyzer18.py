# chess_analyzer_final_uci.py
# A script to analyze PGN chess games and estimate a specific player's rating.
# This version uses an advanced "Weighted Move Quality" score to achieve a
# high R-squared correlation, inspired by the Hindemburg methodology.
#
# KEY FEATURES:
#   1. WEIGHTED MOVE QUALITY: Calculates a nuanced (0.0-1.0) score for each move,
#      where the penalty for centipawn loss is weighted by the position's evaluation.
#   2. FULLY PARALLELIZED: Both major analysis steps for a single game (template
#      creation and model building) are now parallelized to use multiple cores.
#   3. INTERACTIVE & RESUMABLE: Processes one game at a time, prompting the user
#      to continue and saving progress to resume an interrupted session.

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

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
# This should be the path to your compatible Stockfish executable.
ENGINE_PATH = PROJECT_FOLDER / "engines" / "stockfish.exe"
# DEEP ANALYSIS TIME CONTROLS
ANALYSIS_TIME_LIMIT = 6.0  # Time for oracle engine to find the best moves.
MODEL_BUILD_TIME_LIMIT = 6.0 # Time for UCI elo engine to play moves.

# The player name you want to track for an overall average rating.
PLAYER_TO_ANALYZE = "Desjardins373"
# Number of CPU cores to use for the parallel analysis of a single game.
NUM_PROCESSES = 2

# --- SCRIPT SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
if not ENGINE_PATH.is_file():
    logging.error(f"FATAL: Chess engine not found at '{ENGINE_PATH}'")
    sys.exit()

# --- CORE FUNCTIONS ---

def get_best_move_analysis(board, engine):
    """
    Analyzes a position to get the single best move and its evaluation.
    This is the "ground truth" for our quality calculation.
    """
    try:
        info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME_LIMIT), multipv=1)
        # The python-chess library can return a dict for multipv=1, so we normalize it to a list
        if isinstance(info, dict):
            info = [info]
            
        if not info or 'pv' not in info[0] or 'score' not in info[0]:
            return None

        score_obj = info[0]['score'].white()
        if score_obj.is_mate():
            # A very high score for mating, negative if we are being mated
            score = 30000 - abs(score_obj.mate()) if score_obj.mate() > 0 else -30000 + abs(score_obj.mate())
        else:
            score = score_obj.score()

        if score is None: return None
        return {'move': info[0]['pv'][0], 'score': score}

    except Exception as e:
        logging.error(f"An unexpected error occurred in get_best_move_analysis: {repr(e)}")
        return None

def calculate_weighted_move_quality(centipawn_loss, best_move_score):
    """
    Calculates a weighted quality score from 0.0 to 1.0.
    The penalty for centipawn loss is reduced in positions that are already
    overwhelmingly won or lost, as precision matters less.
    """
    # Weighting factor: reduces penalty in non-critical positions
    # The clamp ensures the weight doesn't become negative or too large.
    weight = np.clip(1 - 0.1 * np.log10(1 + abs(best_move_score)), 0.1, 1.0)
    
    # Sigmoid-like function to convert weighted loss to a quality score
    # This creates a smooth curve from 1.0 (perfect) down to 0.0 (blunder)
    quality = 1 / (1 + np.exp(0.04 * (centipawn_loss * weight - 50)))
    return quality

def analyze_position_worker(fen):
    """
    Worker function to get the best move analysis for a single board position.
    """
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        analysis = get_best_move_analysis(board, engine)
    return (fen, analysis)

def test_elo_level_worker(args):
    """
    Worker function for the parallel model builder. Tests a single Elo level.
    """
    elo, move_analyses_for_model = args
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        logging.info(f"  -> Testing at Elo level: {elo}")
        try:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            
            total_quality = 0
            total_moves = 0
            
            for i, (fen, best_move_analysis) in enumerate(move_analyses_for_model.items()):
                if i >= 50: break 
                if not best_move_analysis: continue
                
                board = chess.Board(fen)
                # We need to get the evaluation of the move the engine *actually* plays
                info = engine.analyse(board, chess.engine.Limit(time=MODEL_BUILD_TIME_LIMIT), multipv=1)
                if isinstance(info, dict): info = [info]

                if not info or 'score' not in info[0]: continue
                
                played_move_score = info[0]['score'].white().score(mate_score=30000)
                best_score = best_move_analysis['score']
                
                centipawn_loss = max(0, best_score - played_move_score)
                move_quality = calculate_weighted_move_quality(centipawn_loss, best_score)
                
                total_quality += move_quality
                total_moves += 1
            
            avg_quality = (total_quality / total_moves) if total_moves > 0 else 0
            return (elo, avg_quality)

        except Exception as e:
            logging.error(f"Failed to test at Elo {elo}. Error: {repr(e)}")
            return (elo, None)

def build_model_in_parallel(move_analyses_for_model):
    """
    Builds the performance model by testing Elo levels in parallel.
    """
    logging.info(f"Building performance model in parallel using {len(move_analyses_for_model)} positions...")
    
    elo_levels = np.arange(1350, 3200, 50)
    tasks = [(elo, move_analyses_for_model) for elo in elo_levels]
    
    quality_scores_data = []
    ratings_data = []

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(test_elo_level_worker, tasks)
        for elo, avg_quality in results:
            if avg_quality is not None:
                ratings_data.append(elo)
                quality_scores_data.append(avg_quality)

    if len(ratings_data) < 2:
        return None, None, None, None, None
        
    ratings = np.array(ratings_data).reshape(-1, 1)
    quality_scores = np.array(quality_scores_data)

    model = LinearRegression()
    model.fit(ratings, quality_scores)
    r_squared = model.score(ratings, quality_scores)
    mse = mean_squared_error(quality_scores, model.predict(ratings))
    
    logging.info(f"Model created for this game. R-squared: {r_squared:.4f}, MSE: {mse:.4f}")
    return model, r_squared, mse, ratings, quality_scores

def analyze_player_quality(game, move_analyses):
    """Calculates the average move quality for the target player."""
    total_quality = 0
    moves_counted = 0
    
    board = game.board()
    for move in game.mainline_moves():
        player_name = "White" if board.turn == chess.WHITE else "Black"
        if game.headers.get(player_name) == PLAYER_TO_ANALYZE:
            best_move_analysis = move_analyses.get(board.fen())
            if best_move_analysis:
                # The played move is 'move'. We need its evaluation.
                board.push(move)
                # To get the eval of the move *just made*, we analyze the resulting position
                # and take the score from the opponent's perspective, then flip it.
                with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
                    info = engine.analyse(board, chess.engine.Limit(depth=12))
                    if isinstance(info, dict): info = [info]
                    if info and 'score' in info[0]:
                        # pov(-board.turn) flips the perspective back to the player who moved
                        played_move_score = info[0]['score'].pov(-board.turn).score(mate_score=30000)
                        best_score = best_move_analysis['score']
                        
                        # We need the best score from the current player's perspective
                        best_score_pov = best_score if board.turn == chess.BLACK else -best_score # turn has flipped
                        
                        centipawn_loss = max(0, best_score_pov - played_move_score)
                        move_quality = calculate_weighted_move_quality(centipawn_loss, best_score_pov)
                        
                        total_quality += move_quality
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
    if m == 0: return 0
    return int((quality_score - c) / m)

def save_session(session_data, pgn_path):
    """Saves the current session progress."""
    session_file = pgn_path.with_suffix('.session.json')
    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=4)
    logging.info(f"Session progress saved to {session_file}")

def load_session(pgn_path):
    """Loads a previous session's progress."""
    session_file = pgn_path.with_suffix('.session.json')
    if session_file.exists():
        logging.info(f"Found existing session file: {session_file}")
        with open(session_file, 'r') as f:
            return json.load(f)
    return None

def generate_final_report(session_data):
    """Generates the final PDF report from all completed game data."""
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

        pdf_path = PROJECT_FOLDER / f"Chess_Analysis_Report_FINAL_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(pdf_path)
        logging.info(f"Final report saved to {pdf_path}")
    except Exception as e:
        logging.error(f"Failed to generate final PDF report. Error: {repr(e)}")


def main():
    pgn_path_str = input(f"Enter the full path to your PGN file: ")
    pgn_path = Path(pgn_path_str)
    if not pgn_path.is_file():
        logging.error(f"PGN file not found at {pgn_path}")
        return

    session_data = load_session(pgn_path)
    if session_data:
        logging.info("Resuming previous session.")
    else:
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
        save_session(session_data, pgn_path)

    games_to_process_offsets = session_data['games_to_process_indices']
    total_games_to_process = len(games_to_process_offsets)
    initial_game_count = len(session_data['completed_games_data']) + total_games_to_process
    
    logging.info(f"Found {total_games_to_process} games remaining to analyze for {PLAYER_TO_ANALYZE}.")

    continuous_mode = False
    
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        for i, offset in enumerate(list(games_to_process_offsets)):
            completed_count = len(session_data['completed_games_data'])
            game_num = completed_count + 1
            pgn_file.seek(offset)
            game = chess.pgn.read_game(pgn_file)
            
            logging.info(f"--- Starting Analysis for Game {game_num}/{initial_game_count}: {game.headers.get('White', '?')} vs. {game.headers.get('Black', '?')} ---")

            logging.info("Step 1: Generating best move analyses using all cores...")
            board = game.board()
            fens_to_analyze = [board.fen()]
            for move in game.mainline_moves():
                board.push(move)
                fens_to_analyze.append(board.fen())
            
            move_analyses = {}
            with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
                results = pool.map(analyze_position_worker, fens_to_analyze)
                for fen, analysis in results:
                    if analysis:
                        move_analyses[fen] = analysis
            logging.info("Step 1: Best move analysis complete.")

            player_color = chess.WHITE if game.headers.get("White") == PLAYER_TO_ANALYZE else chess.BLACK
            player_specific_analyses = {fen: an for fen, an in move_analyses.items() if chess.Board(fen).turn == player_color}
            
            model_results = build_model_in_parallel(player_specific_analyses)
            if model_results[0] is None:
                logging.warning(f"Skipping Game {game_num} due to failure in model generation.")
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data, pgn_path)
                continue
            
            model, r_squared, mse, ratings, quality_scores = model_results
            
            plt.figure(figsize=(10, 6))
            plt.scatter(ratings, quality_scores, alpha=0.7, label="Engine Performance (Move Quality)")
            plt.plot(ratings, model.predict(ratings), color='red', linewidth=2, label="Linear Regression Model")
            plt.title(f"Game {game_num}: Engine Rating vs. Move Quality")
            plt.xlabel("Engine Elo Rating (UCI Setting)")
            plt.ylabel("Average Move Quality Score")
            plt.grid(True); plt.legend()
            graph_path = PROJECT_FOLDER / f"performance_graph_game_{game_num}.png"
            plt.savefig(graph_path); plt.close()

            avg_quality, moves_counted = analyze_player_quality(game, move_analyses)
            est_rating = estimate_rating_from_quality(model, avg_quality)
            
            game_data_for_session = {
                'game_num': game_num,
                'white': game.headers.get('White'), 'black': game.headers.get('Black'),
                'r_squared': r_squared,
                'graph_path': str(graph_path),
                'model_coef': model.coef_[0],
                'model_intercept': model.intercept_,
                'avg_quality': avg_quality,
                'moves': moves_counted,
                'estimated_rating': est_rating
            }
            session_data['completed_games_data'].append(game_data_for_session)
            
            session_data['games_to_process_indices'].pop(0)
            save_session(session_data, pgn_path)
            logging.info(f"--- Finished Analysis for Game {game_num}. Results saved. ---")

            if not continuous_mode:
                while True:
                    user_input = input("Continue analysis? (y/n), Mode? (c/b for continuous/batch): ").lower().strip()
                    parts = [p.strip() for p in user_input.split(',')]
                    if len(parts) == 2 and parts[0] in ['y', 'n'] and parts[1] in ['c', 'b']:
                        if parts[0] == 'n':
                            logging.info("Analysis paused by user.")
                            generate_final_report(session_data)
                            return
                        if parts[1] == 'c':
                            continuous_mode = True
                            logging.info("Switching to continuous mode.")
                        break
                    else:
                        print("Invalid input. Please use format: y,c or n,b etc.")
    
    logging.info("All games have been analyzed.")
    generate_final_report(session_data)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
