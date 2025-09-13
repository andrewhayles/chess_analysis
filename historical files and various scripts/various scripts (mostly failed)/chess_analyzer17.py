# chess_analyzer_final_uci.py
# A script to analyze PGN chess games and estimate a specific player's rating.
# This version is a complete redesign inspired by the Hindemburg methodology to
# achieve a high R-squared correlation.
#
# KEY FEATURES:
#   1. DYNAMIC HIT COUNT: Defines a "hit" using a multi-move template and a
#      dynamic centipawn tolerance that adapts to the position's evaluation.
#   2. FULLY PARALLELIZED: Both major analysis steps for a single game (template
#      creation and model building) are now parallelized to use multiple cores.
#   3. INTERACTIVE & RESUMABLE: Processes one game at a time, prompting the user
#      to continue and saving progress to resume an interrupted session.
#   4. DETAILED LOGGING: Creates a running summary log, detailed move-by-move
#      logs for each game, and provides frequent console updates.

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

# --- DYNAMIC HIT TEMPLATE CONFIGURATION ---
# Base centipawn tolerance for a move to be considered part of the template.
BASE_TOLERANCE_CP = 15
# Percentage of the position's evaluation to add to the tolerance.
# e.g., in a +5.00 (500cp) position, tolerance becomes 15 + (500 * 0.05) = 40cp
PERCENTAGE_TOLERANCE = 0.05


# --- SCRIPT SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
if not ENGINE_PATH.is_file():
    logging.error(f"FATAL: Chess engine not found at '{ENGINE_PATH}'")
    sys.exit()

# --- CORE FUNCTIONS ---

def get_move_template(board, engine):
    """
    Analyzes a position to create a 'hit zone' of acceptable moves.
    This is the core of the Hindemburg-inspired method.
    """
    try:
        # Get the top 3 moves to create a richer template
        info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME_LIMIT), multipv=3)
        if not info or len(info) == 0:
            return set()

        top_score_obj = info[0]['score'].white()
        
        # Handle mating lines separately
        if top_score_obj.is_mate():
            template_moves = {info[0]['pv'][0]}
            # Add any other moves that also lead to a mate
            for i in range(1, len(info)):
                if info[i].get('score') and info[i]['score'].white().is_mate():
                    template_moves.add(info[i]['pv'][0])
            return template_moves

        top_move_score = top_score_obj.score()
        if top_move_score is None: return set()

        # The best move is always in the template
        template_moves = {info[0]['pv'][0]}

        # Calculate the dynamic tolerance for this specific position
        dynamic_tolerance = BASE_TOLERANCE_CP + (abs(top_move_score) * PERCENTAGE_TOLERANCE)

        # Add other top moves if they are within the dynamic tolerance
        for i in range(1, len(info)):
            current_score_obj = info[i].get('score')
            if current_score_obj and not current_score_obj.white().is_mate():
                move_score = current_score_obj.white().score()
                if move_score is not None and abs(top_move_score - move_score) <= dynamic_tolerance:
                    template_moves.add(info[i]['pv'][0])
        
        return template_moves
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_move_template: {repr(e)}")
        return set()

def analyze_position_worker(fen):
    """
    Worker function to get the move template for a single board position.
    This is used to parallelize the initial game analysis.
    """
    board = chess.Board(fen)
    # Each worker needs its own engine instance.
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        template = get_move_template(board, engine)
    return (fen, template)

def test_elo_level_worker(args):
    """
    Worker function for the parallel model builder. Tests a single Elo level.
    """
    elo, move_templates_for_model = args
    # Each worker process needs its own engine instance
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        logging.info(f"  -> Testing at Elo level: {elo}")
        try:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            
            hits, total_moves = 0, 0
            for i, (fen, template) in enumerate(move_templates_for_model.items()):
                if i >= 50: break # Cap positions for speed
                board = chess.Board(fen)
                result = engine.play(board, chess.engine.Limit(time=MODEL_BUILD_TIME_LIMIT))
                total_moves += 1
                if result.move in template:
                    hits += 1
            
            hit_rate = (hits / total_moves) if total_moves > 0 else 0
            return (elo, hit_rate)

        except Exception as e:
            logging.error(f"Failed to test at Elo {elo}. Error: {repr(e)}")
            return (elo, None)

def build_model_in_parallel(move_templates_for_model):
    """
    Builds the performance model by testing Elo levels in parallel.
    """
    logging.info(f"Building performance model in parallel using {len(move_templates_for_model)} positions...")
    
    elo_levels = np.arange(1350, 3200, 50)
    
    # Prepare arguments for the worker function
    tasks = [(elo, move_templates_for_model) for elo in elo_levels]
    
    hit_rates_data = []
    ratings_data = []

    # Use a multiprocessing pool to run the tests in parallel
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(test_elo_level_worker, tasks)
        for elo, hit_rate in results:
            if hit_rate is not None:
                ratings_data.append(elo)
                hit_rates_data.append(hit_rate)

    if len(ratings_data) < 2:
        return None, None, None, None, None
        
    ratings = np.array(ratings_data).reshape(-1, 1)
    hit_rates = np.array(hit_rates_data)

    model = LinearRegression()
    model.fit(ratings, hit_rates)
    r_squared = model.score(ratings, hit_rates)
    mse = mean_squared_error(hit_rates, model.predict(ratings))
    
    logging.info(f"Model created for this game. R-squared: {r_squared:.4f}, MSE: {mse:.4f}")
    return model, r_squared, mse, ratings, hit_rates

def analyze_player_hits(game, move_templates):
    """Calculates the hit rate for each player in a game."""
    player_hits = {}
    white_player = game.headers.get("White", "White")
    black_player = game.headers.get("Black", "Black")
    player_hits[white_player] = {'hits': 0, 'moves': 0}
    player_hits[black_player] = {'hits': 0, 'moves': 0}
    
    board = game.board()
    for move in game.mainline_moves():
        fen = board.fen()
        template_for_pos = move_templates.get(fen)
        if template_for_pos:
            player_name = white_player if board.turn == chess.WHITE else black_player
            player_hits[player_name]['moves'] += 1
            if move in template_for_pos:
                player_hits[player_name]['hits'] += 1
        board.push(move)
    return player_hits

def estimate_rating_from_hit_rate(model, hit_rate):
    """Estimates Elo rating from a hit rate using the linear model."""
    m, c = model.coef_[0], model.intercept_
    if m == 0: return 0
    return int((hit_rate - c) / m)

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

def generate_move_log(game_num, game, move_templates):
    """Creates a detailed text log for a single game's move-by-move analysis."""
    log_dir = PROJECT_FOLDER / "move_logs"
    log_dir.mkdir(exist_ok=True) # Ensure the directory exists
    log_path = log_dir / f"move_log_game_{game_num}.txt"

    white_player = game.headers.get("White", "?")
    black_player = game.headers.get("Black", "?")

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Move-by-Move Analysis for Game {game_num}\n")
            f.write(f"White: {white_player}\n")
            f.write(f"Black: {black_player}\n")
            f.write(f"Date: {game.headers.get('Date', '?')}\n")
            f.write("="*40 + "\n\n")

            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                move_num_str = f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
                player_name = white_player if board.turn == chess.WHITE else black_player
                
                fen_before_move = board.fen()
                template = move_templates.get(fen_before_move, set())
                template_str = ", ".join(board.san(m) for m in template) if template else "N/A"
                
                played_move_san = board.san(move)
                is_hit = move in template
                hit_str = "HIT" if is_hit else "MISS"

                f.write(f"Move: {move_num_str} {played_move_san} ({player_name})\n")
                f.write(f"  - Status: {hit_str}\n")
                f.write(f"  - Template Moves: [{template_str}]\n\n")

                board.push(move)
        logging.info(f"Successfully created move-by-move log for Game {game_num}.")
    except Exception as e:
        logging.error(f"Could not create move-by-move log for Game {game_num}. Error: {repr(e)}")

def generate_final_report(session_data):
    """Generates the final PDF report from all completed game data."""
    logging.info("Generating final PDF report...")
    all_games_data = session_data.get('completed_games_data', [])
    if not all_games_data:
        logging.warning("No completed game data to report.")
        return

    # The model objects can't be saved to JSON, so we need to recreate them
    for data in all_games_data:
        if 'model_coef' in data and 'model_intercept' in data:
            model = LinearRegression()
            model.coef_ = np.array([data['model_coef']])
            model.intercept_ = data['model_intercept']
            data['model'] = model

    # Calculate final average data
    player_ratings = [g['estimated_rating'] for g in all_games_data if 'estimated_rating' in g]
    final_average_data = None
    if player_ratings:
        avg_rating = int(np.mean(player_ratings))
        final_average_data = {
            "player_name": PLAYER_TO_ANALYZE,
            "game_count": len(player_ratings),
            "average_rating": avg_rating
        }

    # PDF generation logic remains largely the same
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
            else:
                pdf.multi_cell(w=0, h=5, text="[Graph image not found]")
            
            pdf.ln(5)
            pdf.set_font("Helvetica", '', 10)
            if 'model' in data:
                m, c = data['model'].coef_[0], data['model'].intercept_
                pdf.multi_cell(w=0, h=5, text=f"Model Equation: Hit Rate = {m:.6f} * Rating + {c:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.multi_cell(w=0, h=5, text=f"Model Fit (R-squared): {data['r_squared']:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            
            if 'hit_rate' in data:
                pdf.set_font("Helvetica", 'B', 11)
                pdf.cell(w=0, h=8, text=f"Player: {PLAYER_TO_ANALYZE}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", '', 10)
                pdf.multi_cell(w=0, h=5, text=f"  - Hit Rate: {data['hit_rate']:.2%} ({data['hits']}/{data['moves']} moves)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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

    # --- Session Management ---
    session_data = load_session(pgn_path)
    if session_data:
        logging.info("Resuming previous session.")
    else:
        logging.info("Starting a new analysis session.")
        session_data = {
            'pgn_file': str(pgn_path),
            'games_to_process_indices': [],
            'completed_games_data': [],
            'player_ratings_list': []
        }
        # Read all games and find the ones to process
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
    
    # --- Main Game Loop ---
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        # Create a copy of the list to iterate over, so we can modify the original
        for i, offset in enumerate(list(games_to_process_offsets)):
            # The actual game number is based on how many games are already completed
            completed_count = len(session_data['completed_games_data'])
            game_num = completed_count + 1
            pgn_file.seek(offset)
            game = chess.pgn.read_game(pgn_file)
            
            logging.info(f"--- Starting Analysis for Game {game_num}/{initial_game_count}: {game.headers.get('White', '?')} vs. {game.headers.get('Black', '?')} ---")

            # --- Step 1: Generate move templates IN PARALLEL ---
            logging.info("Step 1: Generating move templates using all cores...")
            board = game.board()
            fens_to_analyze = []
            for move in game.mainline_moves():
                fens_to_analyze.append(board.fen())
                board.push(move)
            
            move_templates = {}
            with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
                results = pool.map(analyze_position_worker, fens_to_analyze)
                for fen, template in results:
                    move_templates[fen] = template
            logging.info("Step 1: Move template generation complete.")

            # *** ADDED BACK MOVE LOG GENERATION ***
            generate_move_log(game_num, game, move_templates)

            # Step 2: Filter templates for the target player's moves
            player_color = chess.WHITE if game.headers.get("White") == PLAYER_TO_ANALYZE else chess.BLACK
            player_specific_templates = {fen: template for fen, template in move_templates.items() if chess.Board(fen).turn == player_color}
            
            # Step 3: Build the performance model in parallel
            model_results = build_model_in_parallel(player_specific_templates)
            if model_results[0] is None:
                logging.warning(f"Skipping Game {game_num} due to failure in model generation.")
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data, pgn_path)
                continue
            
            model, r_squared, mse, ratings, hit_rates = model_results
            
            # Step 4: Plot the model
            plt.figure(figsize=(10, 6))
            plt.scatter(ratings, hit_rates, alpha=0.7, label="Engine Performance (Hit Rate)")
            plt.plot(ratings, model.predict(ratings), color='red', linewidth=2, label="Linear Regression Model")
            plt.title(f"Game {game_num}: Engine Rating vs. Hit Rate")
            plt.xlabel("Engine Elo Rating (UCI Setting)")
            plt.ylabel("Hit Rate")
            plt.grid(True); plt.legend()
            graph_path = PROJECT_FOLDER / f"performance_graph_game_{game_num}.png"
            plt.savefig(graph_path); plt.close()

            # Step 5: Analyze human player
            player_results = analyze_player_hits(game, move_templates)
            player_data = player_results.get(PLAYER_TO_ANALYZE, {'hits': 0, 'moves': 0})
            hit_rate = (player_data['hits'] / player_data['moves']) if player_data['moves'] > 0 else 0
            est_rating = estimate_rating_from_hit_rate(model, hit_rate)
            
            # Step 6: Store results for this game
            game_data_for_session = {
                'game_num': game_num,
                'white': game.headers.get('White'), 'black': game.headers.get('Black'),
                'r_squared': r_squared,
                'graph_path': str(graph_path),
                'model_coef': model.coef_[0],
                'model_intercept': model.intercept_,
                'hit_rate': hit_rate,
                'hits': player_data['hits'],
                'moves': player_data['moves'],
                'estimated_rating': est_rating
            }
            session_data['completed_games_data'].append(game_data_for_session)
            session_data['player_ratings_list'].append(est_rating)
            
            # Remove the processed game from the list of games to do
            session_data['games_to_process_indices'].pop(0)
            save_session(session_data, pgn_path)
            logging.info(f"--- Finished Analysis for Game {game_num}. Results saved. ---")

            # --- Interactive Prompt ---
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