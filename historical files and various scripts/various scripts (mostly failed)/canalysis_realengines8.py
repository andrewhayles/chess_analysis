# chess_analyzer_multi_engine.py
# A script to analyze PGN chess games and estimate a specific player's rating.
# This version uses a sophisticated multi-engine approach inspired by the
# Hindemburg methodology to achieve a high R-squared correlation.
#
# KEY FEATURES:
#   1. MAXIMUM STABILITY: Uses a "one task, one process" model with a non-blocking
#      results queue (imap_unordered) and generous per-task timeouts to make the system
#      immune to individual engine crashes and hangs.
#   2. ADAPTIVE ORACLE PANEL: Intelligently queries engines, trying for multiple
#      best moves but gracefully falling back to a single best move if an
#      engine does not support MultiPV.
#   3. CSV-DRIVEN ENGINES: Reads a list of engines from a 'real_engines.csv' file.
#   4. ISOLATED OUTPUT: Saves all logs, graphs, and reports for a session into a
#      unique, timestamped directory.
#   5. INTERACTIVE & RESUMABLE: Processes one game at a time, prompting the user
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
import csv
from collections import defaultdict

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
ENGINES_CSV_PATH = PROJECT_FOLDER / "real_engines.csv"
# DEEP ANALYSIS TIME CONTROLS
ANALYSIS_TIME_LIMIT = 6.0
MODEL_BUILD_TIME_LIMIT = 6.0

PLAYER_TO_ANALYZE = "Desjardins373"
NUM_ORACLE_ENGINES = 4 # The first N engines in the CSV are used as the oracle panel

# --- SCRIPT SETUP ---
# All outputs for this run will be stored in a session-specific folder
SESSION_TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
SESSION_FOLDER = PROJECT_FOLDER / f"real_engines_run_{SESSION_TIMESTAMP}"
SESSION_FOLDER.mkdir(exist_ok=True)

# Setup logging to file and console
log_file_path = SESSION_FOLDER / 'analysis_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

if not ENGINES_CSV_PATH.is_file():
    logging.error(f"FATAL: Engines CSV file not found at '{ENGINES_CSV_PATH}'")
    sys.exit()

# --- ISOLATED WORKER FUNCTIONS ---

def oracle_analysis_worker(args):
    """
    Worker for a single oracle analysis task. Opens an engine, performs
    one analysis, and closes the engine. This provides maximum stability.
    """
    fen, engine_info = args
    board = chess.Board(fen)
    top_moves = []
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            try:
                # First, try to get the top 3 moves (MultiPV)
                info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME_LIMIT), multipv=3)
                top_moves = [item['pv'][0] for item in info if 'pv' in item and item['pv']]
            except chess.engine.EngineError:
                # If MultiPV fails, fall back to a single best move
                logging.warning(f"Engine {engine_info['name']} does not support MultiPV. Falling back.")
                info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME_LIMIT), multipv=1)
                if isinstance(info, dict) and 'pv' in info and info['pv']:
                    top_moves = [info['pv'][0]]
    except Exception as e:
        logging.error(f"Worker for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
        return (fen, [], engine_info) # Return empty list on failure
    
    return (fen, top_moves, engine_info)

def model_engine_worker(args):
    """
    Worker for a single model engine task. Calculates the hit rate for one
    engine across all positions.
    """
    engine_info, player_templates = args
    hits = 0
    moves = 0
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            logging.info(f"  -> Testing model engine: {engine_info['name']}")
            for fen, template in player_templates.items():
                if not template: continue
                board = chess.Board(fen)
                result = engine.play(board, chess.engine.Limit(time=MODEL_BUILD_TIME_LIMIT))
                if result.move:
                    moves += 1
                    if result.move in template:
                        hits += 1
    except Exception as e:
        logging.error(f"Worker for model engine {engine_info['name']} failed: {repr(e)}")
        return (engine_info, None)

    hit_rate = (hits / moves) if moves > 0 else 0
    return (engine_info, hit_rate)


# --- MAIN ANALYSIS FUNCTIONS ---

def get_weighted_template(game, oracle_engines, num_processes):
    """
    Analyzes all relevant positions in a game with the oracle panel
    and creates a plurality-vote template.
    """
    logging.info(f"Generating weighted template with {len(oracle_engines)} oracle engines...")
    
    fens_to_analyze = []
    board = game.board()
    player_color = chess.WHITE if game.headers.get("White") == PLAYER_TO_ANALYZE else chess.BLACK
    for move in game.mainline_moves():
        if board.turn == player_color:
            fens_to_analyze.append(board.fen())
        board.push(move)
    
    tasks = [(fen, engine_info) for fen in fens_to_analyze for engine_info in oracle_engines]

    position_votes = defaultdict(lambda: defaultdict(int))
    position_voters = defaultdict(list)

    pool = multiprocessing.Pool(processes=num_processes)
    try:
        # Use imap_unordered for non-blocking results
        result_iterator = pool.imap_unordered(oracle_analysis_worker, tasks)
        
        for i in range(len(tasks)):
            try:
                # Apply a generous timeout to getting each result
                fen, top_moves, engine_info = result_iterator.next(timeout=ANALYSIS_TIME_LIMIT * 5)
                if top_moves:
                    position_voters[fen].append({'moves': top_moves, 'engine': engine_info})
                    for j, move in enumerate(top_moves):
                        position_votes[fen][move] += (3 - j)
            except multiprocessing.TimeoutError:
                logging.warning("Timeout waiting for an oracle worker to complete. This is likely due to an unstable engine. Skipping task.")
            except Exception as e:
                logging.error(f"An error occurred while processing oracle results: {repr(e)}")
    finally:
        pool.close()
        pool.join()


    final_templates = {}
    for fen, votes in position_votes.items():
        if votes:
            max_votes = max(votes.values())
            top_voted_moves = [move for move, count in votes.items() if count == max_votes]
            
            if len(top_voted_moves) == 1:
                final_templates[fen] = {top_voted_moves[0]}
            else:
                best_tied_move, highest_rating = None, -1
                for move in top_voted_moves:
                    for voter_info in position_voters[fen]:
                        if move in voter_info['moves'] and voter_info['engine']['rating'] > highest_rating:
                            highest_rating = voter_info['engine']['rating']
                            best_tied_move = move
                if best_tied_move:
                    final_templates[fen] = {best_tied_move}
    
    logging.info("Weighted template generation complete.")
    return final_templates

def build_model_with_real_engines(player_specific_templates, model_engines, num_processes):
    """
    Builds the performance model using the panel of modeling engines.
    """
    logging.info(f"Building performance model with {len(model_engines)} real engines...")
    
    tasks = [(engine_info, player_specific_templates) for engine_info in model_engines]
    
    ratings_data, hit_rates_data = [], []
    pool = multiprocessing.Pool(processes=num_processes)
    try:
        result_iterator = pool.imap_unordered(model_engine_worker, tasks)
        
        for i in range(len(tasks)):
            try:
                # Apply a very generous timeout
                engine_info, hit_rate = result_iterator.next(timeout=MODEL_BUILD_TIME_LIMIT * (len(player_specific_templates) + 5))
                if hit_rate is not None:
                    ratings_data.append(engine_info['rating'])
                    hit_rates_data.append(hit_rate)
            except multiprocessing.TimeoutError:
                logging.warning("Timeout waiting for a model worker to complete. This is likely due to an unstable engine. Skipping engine.")
            except Exception as e:
                logging.error(f"An error occurred while processing model results: {repr(e)}")
    finally:
        pool.close()
        pool.join()


    if len(ratings_data) < 2: return None, None, None, None

    ratings = np.array(ratings_data).reshape(-1, 1)
    hit_rates = np.array(hit_rates_data)
    model = LinearRegression()
    model.fit(ratings, hit_rates)
    r_squared = model.score(ratings, hit_rates)
    
    logging.info(f"Model created. R-squared: {r_squared:.4f}")
    return model, r_squared, ratings, hit_rates

# --- UTILITY AND REPORTING FUNCTIONS ---

def analyze_player_hits(game, templates):
    """Calculates the hit rate for the target player."""
    hits, moves = 0, 0
    board = game.board()
    player_color = chess.WHITE if game.headers.get("White") == PLAYER_TO_ANALYZE else chess.BLACK
    for move in game.mainline_moves():
        if board.turn == player_color:
            template = templates.get(board.fen())
            if template:
                moves += 1
                if move in template:
                    hits += 1
        board.push(move)
    return hits, moves

def estimate_rating_from_hit_rate(model, hit_rate):
    m, c = model.coef_[0], model.intercept_
    if m <= 0: return 0
    return int((hit_rate - c) / m)

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

def generate_move_log(game_num, game, templates, session_folder):
    """Creates a detailed text log for a single game's move-by-move analysis."""
    log_dir = session_folder / "move_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"move_log_game_{game_num}.txt"

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Move-by-Move Analysis for Game {game_num}\n")
            f.write(f"White: {game.headers.get('White', '?')}\nBlack: {game.headers.get('Black', '?')}\n")
            f.write("="*40 + "\n\n")

            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                move_num_str = f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}"
                player_name = game.headers.get("White") if board.turn == chess.WHITE else game.headers.get("Black")
                
                fen_before_move = board.fen()
                template = templates.get(fen_before_move, set())
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
    
    if len(all_engines) <= NUM_ORACLE_ENGINES:
        logging.error("Not enough engines in CSV for both oracle and model panels.")
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

            # Step 1: Generate the weighted template for this game
            player_templates = get_weighted_template(game, oracle_engines, len(oracle_engines))
            generate_move_log(game_num, game, player_templates, SESSION_FOLDER)

            # Step 2: Build the performance model using the model engines
            model_results = build_model_with_real_engines(player_templates, model_engines, len(model_engines))
            if model_results[0] is None:
                logging.warning(f"Skipping Game {game_num} due to failure in model generation.")
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data, pgn_path, SESSION_FOLDER)
                continue
            
            model, r_squared, ratings, hit_rates = model_results
            
            # Step 3: Plotting
            plt.figure(figsize=(10, 6))
            plt.scatter(ratings, hit_rates, alpha=0.7, label="Engine Performance (Hit Rate)")
            plt.plot(ratings, model.predict(ratings), color='red', linewidth=2, label="Linear Regression Model")
            plt.title(f"Game {game_num}: Engine Rating vs. Hit Rate")
            plt.xlabel("Engine Elo Rating (from CSV)")
            plt.ylabel("Hit Rate")
            plt.grid(True); plt.legend()
            graph_path = SESSION_FOLDER / f"performance_graph_game_{game_num}.png"
            plt.savefig(graph_path); plt.close()

            # Step 4: Analyze human player
            hits, moves = analyze_player_hits(game, player_templates)
            hit_rate = (hits / moves) if moves > 0 else 0
            est_rating = estimate_rating_from_hit_rate(model, hit_rate)
            
            # Step 5: Save results
            game_data_for_session = {
                'game_num': game_num, 'white': game.headers.get('White'), 'black': game.headers.get('Black'),
                'r_squared': r_squared, 'graph_path': str(graph_path),
                'model_coef': model.coef_[0], 'model_intercept': model.intercept_,
                'hit_rate': hit_rate, 'hits': hits, 'moves': moves,
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
