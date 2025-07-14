# chess_analyzer_final_uci.py
# A script to analyze PGN chess games and estimate a specific player's rating.
# This ADVANCED version implements methods inspired by academic research:
#   1. It calculates player ACCURACY based on CENTIPAWN LOSS, not just hit/miss.
#   2. It uses a HYBRID MODELING approach, automatically selecting the best fit from
#      Linear, Polynomial, and a ROBUST (Tukey's Biweight) regression model to
#      achieve a higher and more reliable R-squared value.

import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm # Used for Robust Linear Modeling
import matplotlib
matplotlib.use('Agg') # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from pathlib import Path
import datetime
import logging
import multiprocessing
import os

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
# This should be the path to your compatible Stockfish executable.
ENGINE_PATH = PROJECT_FOLDER / "engines" / "stockfish.exe"
ANALYSIS_TIME_LIMIT = 6  # Time for oracle engine to find the best move and its score.
MODEL_BUILD_TIME_LIMIT = 2 # Time for UCI elo engine to play moves.

# The player name you want to track for an overall average rating.
PLAYER_TO_ANALYZE = "Desjardins373"
# Number of CPU cores to use for analysis.
NUM_PROCESSES = 2
# The degree of the polynomial model to use (2=quadratic, 3=cubic).
POLYNOMIAL_DEGREE = 2
# This constant helps convert centipawn loss into an accuracy score (0-1).
# A higher value means mistakes are penalized more harshly.
ACCURACY_SENSITIVITY = 0.03

# --- SCRIPT SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
if not ENGINE_PATH.is_file():
    logging.error(f"FATAL: Chess engine not found at '{ENGINE_PATH}'")
    exit()

# --- CORE FUNCTIONS ---

def get_move_analysis(board, engine):
    """
    Analyzes a board position to get the best move and its centipawn score.
    Returns a dictionary containing the best move and its score.
    """
    try:
        # We only need the top move for centipawn loss calculation.
        info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME_LIMIT), multipv=1)
        if not info or 'pv' not in info[0] or 'score' not in info[0]:
            return None
        
        score = info[0]['score'].white()
        # If it's a mate, the loss is effectively infinite for any non-mating move.
        # We'll handle this by giving it a very high, but finite, score.
        if score.is_mate():
            # Positive for white's advantage, negative for black's
            mate_score = 30000 - abs(score.mate()) if score.mate() > 0 else -30000 + abs(score.mate())
            return {'move': info[0]['pv'][0], 'score': mate_score}

        return {'move': info[0]['pv'][0], 'score': score.score()}
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_move_analysis: {e}")
        return None

def calculate_accuracy(centipawn_loss):
    """
    Converts centipawn loss to an accuracy score between 0 and 1.
    Uses an exponential decay function.
    """
    return np.exp(-ACCURACY_SENSITIVITY * (centipawn_loss / 100.0))

def build_performance_models(engine, move_analyses_for_model):
    """
    Generates performance data using the new accuracy metric and creates three
    different models, returning the one with the best fit.
    """
    logging.info(f"Building performance models using {len(move_analyses_for_model)} positions...")
    
    elo_levels = np.arange(1300, 3200, 50)
    ratings_data = []
    accuracies_data = []

    for elo in elo_levels:
        # ADDED: This log provides live status updates during the longest part of the analysis.
        logging.info(f"  -> Testing at Elo level: {elo}")
        try:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            
            total_accuracy_score = 0
            total_moves = 0
            
            for i, (fen, best_move_analysis) in enumerate(move_analyses_for_model.items()):
                if i >= 50: break 
                board = chess.Board(fen)
                
                # Get the engine's move and score for that move
                result_info = engine.analyse(board, chess.engine.Limit(time=MODEL_BUILD_TIME_LIMIT), multipv=1)
                if not result_info: continue
                
                played_move_score = result_info[0]['score'].white().score(mate_score=30000)
                best_score = best_move_analysis['score']
                
                # Centipawn loss is the difference in score from the best move.
                # It's always positive or zero.
                centipawn_loss = max(0, best_score - played_move_score)
                
                total_accuracy_score += calculate_accuracy(centipawn_loss)
                total_moves += 1

            avg_accuracy = (total_accuracy_score / total_moves) if total_moves > 0 else 0
            ratings_data.append(elo)
            accuracies_data.append(avg_accuracy)

        except Exception as e:
            logging.error(f"Failed to test at Elo {elo}. Error: {e}")
            continue

    engine.configure({"UCI_LimitStrength": False})

    if len(ratings_data) < 2:
        return None, None, None, None, None, None
        
    ratings = np.array(ratings_data).reshape(-1, 1)
    accuracies = np.array(accuracies_data)

    # --- Model Fitting ---
    # We will now fit THREE models and choose the best one.
    models = {}

    # 1. Linear Model
    linear_model = LinearRegression()
    linear_model.fit(ratings, accuracies)
    models['Linear'] = (linear_model, linear_model.score(ratings, accuracies))

    # 2. Polynomial Model
    poly_model = make_pipeline(PolynomialFeatures(degree=POLYNOMIAL_DEGREE), LinearRegression())
    poly_model.fit(ratings, accuracies)
    models['Polynomial'] = (poly_model, poly_model.score(ratings, accuracies))
    
    # 3. Robust Model (Tukey's Biweight)
    X = sm.add_constant(ratings) # statsmodels requires an explicit constant
    # *** FIXED THE ERROR HERE ***
    robust_model = sm.RLM(accuracies, X, M=sm.robust.norms.TukeyBiweight()).fit()
    # R-squared for RLM needs to be calculated manually
    ss_total = np.sum((accuracies - np.mean(accuracies))**2)
    ss_resid = np.sum(robust_model.resid**2)
    robust_r_squared = 1 - (ss_resid / ss_total)
    models['Robust (Tukey)'] = (robust_model, robust_r_squared)

    # Find the best model based on R-squared
    best_model_type = max(models, key=lambda k: models[k][1])
    final_model, final_r_squared = models[best_model_type]
    
    # For MSE calculation, we need a consistent predict method
    if best_model_type == 'Robust (Tukey)':
        predictions = final_model.predict(X)
    else:
        predictions = final_model.predict(ratings)
    final_mse = mean_squared_error(accuracies, predictions)
    
    logging.info(f"Model comparison: Linear R²={models['Linear'][1]:.4f}, "
                 f"Polynomial R²={models['Polynomial'][1]:.4f}, "
                 f"Robust R²={models['Robust (Tukey)'][1]:.4f}")
    logging.info(f"Best model selected: {best_model_type}. Final R-squared: {final_r_squared:.4f}")
    
    return final_model, final_r_squared, final_mse, ratings, accuracies, best_model_type


def analyze_player_accuracy(game, move_analyses, engine):
    """Calculates the accuracy score for each player in a game."""
    player_accuracies = {}
    white_player = game.headers.get("White", "White")
    black_player = game.headers.get("Black", "Black")
    player_accuracies[white_player] = {'total_accuracy': 0, 'moves': 0}
    player_accuracies[black_player] = {'total_accuracy': 0, 'moves': 0}
    
    board = game.board()
    for move in game.mainline_moves():
        fen = board.fen()
        best_move_analysis = move_analyses.get(fen)
        
        if best_move_analysis:
            player_name = white_player if board.turn == chess.WHITE else black_player
            
            # We need to evaluate the player's actual move to get its score
            board.push(move)
            info = engine.analyse(board, chess.engine.Limit(depth=10)) # Quick eval of the resulting position
            board.pop() # Go back
            
            if info:
                # The score is from the perspective of the player who just moved.
                # So we need to flip the sign if it's black's turn.
                current_player_pov_score = info[0]['score'].pov(board.turn).score(mate_score=30000)
                
                # The best score also needs to be from the current player's perspective
                best_score_pov = best_move_analysis['score'] if board.turn == chess.WHITE else -best_move_analysis['score']
                
                centipawn_loss = max(0, best_score_pov - current_player_pov_score)
                accuracy = calculate_accuracy(centipawn_loss)
                
                player_accuracies[player_name]['total_accuracy'] += accuracy
                player_accuracies[player_name]['moves'] += 1

        board.push(move)
    return player_accuracies

def estimate_rating_from_accuracy(model, accuracy_score, model_type):
    """Estimates Elo rating from an accuracy score using the chosen model."""
    elo_search_range = np.arange(1000, 3500, 10).reshape(-1, 1)

    if model_type == 'Robust (Tukey)':
        # statsmodels needs the constant added for prediction
        X_search = sm.add_constant(elo_search_range)
        predicted_accuracies = model.predict(X_search)
    else: # Linear or Polynomial
        predicted_accuracies = model.predict(elo_search_range)
        
    closest_index = np.argmin(np.abs(predicted_accuracies - accuracy_score))
    return elo_search_range[closest_index][0]


def process_game(game_tuple):
    """
    Worker function to process a single chess game.
    """
    game, game_num = game_tuple

    try:
        with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
            logging.info(f"Processing Game {game_num}: {game.headers.get('White', '?')} vs. {game.headers.get('Black', '?')}")

            # Step 1: Generate move analyses (best move + score) for all positions.
            move_analyses = {}
            board = game.board()
            for move in game.mainline_moves():
                analysis = get_move_analysis(board, engine)
                if analysis:
                    move_analyses[board.fen()] = analysis
                board.push(move)

            # Step 2: Filter analyses for only the target player's positions.
            player_color = chess.WHITE if game.headers.get("White") == PLAYER_TO_ANALYZE else chess.BLACK
            player_specific_analyses = {}
            board = game.board()
            for move in game.mainline_moves():
                if board.turn == player_color:
                    if board.fen() in move_analyses:
                        player_specific_analyses[board.fen()] = move_analyses[board.fen()]
                board.push(move)

            # Step 3: Build performance models.
            model_results = build_performance_models(engine, player_specific_analyses)
            if model_results[0] is None:
                logging.warning(f"Skipping Game {game_num} due to failure in model generation.")
                return None

            model, r_squared, mse, ratings, accuracies, model_type = model_results
            
            # Step 4: Plotting
            plt.figure(figsize=(10, 6))
            plt.scatter(ratings, accuracies, alpha=0.6, label="Engine Performance (Accuracy)")
            
            sorted_ratings_for_plot = np.sort(ratings, axis=0)
            if model_type == 'Robust (Tukey)':
                X_plot = sm.add_constant(sorted_ratings_for_plot)
                plt.plot(sorted_ratings_for_plot, model.predict(X_plot), color='red', linewidth=2, label=f"{model_type} Model")
            else:
                plt.plot(sorted_ratings_for_plot, model.predict(sorted_ratings_for_plot), color='red', linewidth=2, label=f"{model_type} Model")

            plt.title(f"Game {game_num}: Engine Rating vs. Accuracy ({game.headers.get('White')} vs. {game.headers.get('Black')})")
            plt.xlabel("Engine Elo Rating (UCI Setting)")
            plt.ylabel("Average Accuracy Score")
            plt.grid(True); plt.legend()
            graph_path = PROJECT_FOLDER / f"performance_graph_game_{game_num}.png"
            plt.savefig(graph_path); plt.close()

            # Step 5: Analyze human player's accuracy
            player_results = analyze_player_accuracy(game, move_analyses, engine)
            
            # Step 6: Package results
            game_data = {
                'game_num': game_num, 'white': game.headers.get('White'), 'black': game.headers.get('Black'),
                'model': model, 'r_squared': r_squared, 'mse': mse, 'model_type': model_type,
                'graph_path': str(graph_path), 'player_results': player_results
            }
            logging.info(f"Finished processing Game {game_num}.")
            return game_data

    except Exception as e:
        logging.error(f"A critical error occurred in worker for game {game_num}: {e}")
        return None

# --- Main Execution & Reporting (Largely unchanged, but adapted for new metrics) ---

def update_running_log(log_path, game_num, game_data, running_avg_data):
    """Appends the latest game analysis and running average to the log file."""
    try:
        write_header = not log_path.exists()
        with open(log_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("Chess Performance Analysis - Live Log\n")
                f.write(f"Analysis started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Target Player: {PLAYER_TO_ANALYZE}\n")
                f.write("="*50 + "\n")

            f.write(f"\n--- Analysis for Game {game_num}: {game_data['white']} vs. {game_data['black']} ---\n")
            f.write(f"  Model Used: {game_data['model_type']} (R² = {game_data['r_squared']:.4f})\n")
            
            if PLAYER_TO_ANALYZE in game_data['player_results']:
                results = game_data['player_results'][PLAYER_TO_ANALYZE]
                avg_accuracy = (results['total_accuracy'] / results['moves']) if results['moves'] > 0 else 0
                est_rating = estimate_rating_from_accuracy(game_data['model'], avg_accuracy, game_data['model_type'])
                f.write(f"  Player: {PLAYER_TO_ANALYZE}\n")
                f.write(f"    - Average Accuracy: {avg_accuracy:.2%}\n")
                f.write(f"    - Estimated Game Rating: {est_rating} Elo\n")
            
            if running_avg_data:
                f.write("\n--- Running Average Update ---\n")
                f.write(f"  Player: {running_avg_data['player_name']}\n")
                f.write(f"  Games Analyzed: {running_avg_data['game_count']}\n")
                f.write(f"  Current Average Rating: {running_avg_data['average_rating']} Elo\n")
            f.write("="*50 + "\n")
    except Exception as e:
        logging.error(f"Could not update running log file. Error: {e}")

def generate_report_pdf(all_games_data, average_rating_data):
    """Generates a PDF report summarizing the analysis."""
    logging.info("Generating final PDF report...")
    try:
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(w=0, h=10, text="Chess Performance Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(w=0, h=5, text=f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)

        all_games_data.sort(key=lambda x: x['game_num'])

        for i, data in enumerate(all_games_data):
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(w=0, h=10, text=f"Analysis for Game {data['game_num']}: {data['white']} vs. {data['black']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            if Path(data['graph_path']).is_file():
                pdf.image(data['graph_path'], w=180)
            else:
                pdf.multi_cell(w=0, h=5, text="[Graph image not found]")
            
            pdf.ln(5)

            pdf.set_font("Helvetica", '', 10)
            pdf.multi_cell(w=0, h=5, text=f"Model Used: {data['model_type']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.multi_cell(w=0, h=5, text=f"Model Fit (R-squared): {data['r_squared']:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            
            if PLAYER_TO_ANALYZE in data['player_results']:
                results = data['player_results'][PLAYER_TO_ANALYZE]
                pdf.set_font("Helvetica", 'B', 11)
                pdf.cell(w=0, h=8, text=f"Player: {PLAYER_TO_ANALYZE}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", '', 10)
                avg_accuracy = (results['total_accuracy'] / results['moves']) if results['moves'] > 0 else 0
                est_rating = estimate_rating_from_accuracy(data['model'], avg_accuracy, data['model_type'])
                
                pdf.multi_cell(w=0, h=5, text=f"  - Average Accuracy: {avg_accuracy:.2%}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.multi_cell(w=0, h=5, text=f"  - Estimated Game Rating: {est_rating} Elo", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            pdf.ln(5)
            if (i + 1) < len(all_games_data):
                 pdf.add_page()

        if average_rating_data:
            if len(all_games_data) > 0: pdf.add_page()
            pdf.set_font("Helvetica", 'B', 16)
            pdf.cell(w=0, h=10, text="Overall Performance Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            pdf.ln(10)
            
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(w=0, h=10, text=f"Player: {average_rating_data['player_name']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            pdf.set_font("Helvetica", '', 10)
            pdf.multi_cell(w=0, h=5, text=f"Analyzed across {average_rating_data['game_count']} games.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.multi_cell(w=0, h=5, text=f"Average Estimated Performance Rating: {average_rating_data['average_rating']} Elo", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
        pdf_path = PROJECT_FOLDER / f"Chess_Analysis_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(pdf_path)
        logging.info(f"Report saved to {pdf_path}")
    except Exception as e:
        logging.error(f"Failed to generate PDF report. Error: {e}")
        logging.error("However, the text summary file should be available.")


def main():
    pgn_path_str = input(f"Enter the full path to your PGN file (or press Enter to use a sample): ")
    pgn_path = Path(pgn_path_str) if pgn_path_str else PROJECT_FOLDER / "sample_game.pgn"
    if not pgn_path.is_file():
        if not pgn_path_str:
            sample_pgn_content = """
[Event "Sample Game"]
[Site "?"]
[Date "2025.07.03"]
[Round "?"]
[White "Player A"]
[Black "Desjardins373"]
[Result "1-0"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0
"""
            with open(pgn_path, "w") as f:
                f.write(sample_pgn_content)
            logging.info(f"Using sample PGN file: {pgn_path}")
        else:
            logging.error(f"PGN file not found at {pgn_path}")
            return

    games_to_process = []
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        game_num = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None: break
            if PLAYER_TO_ANALYZE in (game.headers.get("White"), game.headers.get("Black")):
                game_num += 1
                games_to_process.append((game, game_num))

    if not games_to_process:
        logging.error(f"No games found for player '{PLAYER_TO_ANALYZE}' in the PGN file.")
        return

    logging.info(f"Found {len(games_to_process)} games to analyze for {PLAYER_TO_ANALYZE}.")
    logging.info(f"Starting parallel analysis with {NUM_PROCESSES} processor cores...")

    all_games_data = []
    player_ratings_list = []
    final_average_data = None
    log_path = PROJECT_FOLDER / "running_analysis_log.txt"
    if log_path.exists():
        log_path.unlink()

    try:
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            results_iterator = pool.imap_unordered(process_game, games_to_process)

            for game_data in results_iterator:
                if game_data is None:
                    continue
                
                all_games_data.append(game_data)
                running_avg_data = None
                if PLAYER_TO_ANALYZE in game_data['player_results']:
                    player_data = game_data['player_results'][PLAYER_TO_ANALYZE]
                    if player_data['moves'] > 0:
                        avg_accuracy = player_data['total_accuracy'] / player_data['moves']
                        est_rating = estimate_rating_from_accuracy(game_data['model'], avg_accuracy, game_data['model_type'])
                        player_ratings_list.append(est_rating)
                        
                        avg_rating = int(np.mean(player_ratings_list))
                        n = len(player_ratings_list)
                        
                        running_avg_data = {
                            "player_name": PLAYER_TO_ANALYZE, "game_count": n,
                            "average_rating": avg_rating,
                        }
                
                update_running_log(log_path, game_data['game_num'], game_data, running_avg_data)
                logging.info(f"Live log updated for Game {game_data['game_num']}.")
                final_average_data = running_avg_data

    except Exception as e:
        logging.error(f"A critical error occurred during the multiprocessing phase: {e}")
        return

    if not all_games_data:
        logging.error("No games were successfully analyzed after parallel processing.")
        return
    
    generate_report_pdf(all_games_data, final_average_data)
    logging.info("Analysis complete.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
