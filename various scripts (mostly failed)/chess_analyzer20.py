# chess_analyzer_final_uci.py
# A script to analyze PGN chess games and estimate a specific player's rating.
# This version uses a robust, persistent engine pool architecture and a
# DYNAMIC ANALYSIS TIME (based on search depth) to ensure stability and quality.
#
# KEY FEATURES:
#   1. DEPTH-BASED ANALYSIS: Uses a fixed search depth instead of a fixed time
#      to allow the engine to dynamically allocate time based on position complexity.
#   2. STABLE ENGINE POOL: Creates a single, persistent pool of engine processes
#      per game to handle all analysis tasks, preventing deadlocks.
#   3. DYNAMIC HIT COUNT: Defines a "hit" using a multi-move template and a
#      dynamic centipawn tolerance that adapts to the position's evaluation.
#   4. INTERACTIVE & RESUMABLE: Processes one game at a time, prompting the user
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
import queue

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
# This should be the path to your compatible Stockfish executable.
ENGINE_PATH = PROJECT_FOLDER / "engines" / "stockfish.exe"

# --- NEW DEPTH-BASED ANALYSIS CONTROLS ---
ORACLE_ANALYSIS_DEPTH = 20  # Depth for the oracle engine to find best moves.
MODEL_BUILD_DEPTH = 12    # Depth for the Elo-limited engines to play moves.

PLAYER_TO_ANALYZE = "Desjardins373"
NUM_PROCESSES = 2

# --- DYNAMIC HIT TEMPLATE CONFIGURATION ---
BASE_TOLERANCE_CP = 15
PERCENTAGE_TOLERANCE = 0.05

# --- SCRIPT SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
if not ENGINE_PATH.is_file():
    logging.error(f"FATAL: Chess engine not found at '{ENGINE_PATH}'")
    sys.exit()

# --- ENGINE POOL WORKER FUNCTION ---

def engine_worker(task_queue, result_queue):
    """
    A persistent worker that keeps an engine instance open.
    It pulls tasks from the queue and puts results in another queue.
    """
    try:
        with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
            while True:
                try:
                    task_type, data = task_queue.get()
                    if task_type == "STOP":
                        break

                    if task_type == "analyze_position":
                        fen = data
                        board = chess.Board(fen)
                        # --- Get Template using DEPTH limit ---
                        info = engine.analyse(board, chess.engine.Limit(depth=ORACLE_ANALYSIS_DEPTH), multipv=3)
                        if not info or len(info) == 0:
                            result_queue.put(("analyze_position", fen, set()))
                            continue

                        top_score_obj = info[0]['score'].white()
                        if top_score_obj.is_mate():
                            template = {info[0]['pv'][0]}
                            for i in range(1, len(info)):
                                if info[i].get('score') and info[i]['score'].white().is_mate():
                                    template.add(info[i]['pv'][0])
                        else:
                            top_move_score = top_score_obj.score()
                            if top_move_score is None:
                                template = set()
                            else:
                                template = {info[0]['pv'][0]}
                                dynamic_tolerance = BASE_TOLERANCE_CP + (abs(top_move_score) * PERCENTAGE_TOLERANCE)
                                for i in range(1, len(info)):
                                    current_score_obj = info[i].get('score')
                                    if current_score_obj and not current_score_obj.white().is_mate():
                                        move_score = current_score_obj.white().score()
                                        if move_score is not None and abs(top_move_score - move_score) <= dynamic_tolerance:
                                            template.add(info[i]['pv'][0])
                        result_queue.put(("analyze_position", fen, template))

                    elif task_type == "test_elo":
                        elo, move_templates = data
                        logging.info(f"  -> Testing at Elo level: {elo}")
                        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
                        hits, total_moves = 0, 0
                        for i, (fen, template) in enumerate(move_templates.items()):
                            if i >= 50: break
                            board = chess.Board(fen)
                            # --- Play move using DEPTH limit ---
                            result = engine.play(board, chess.engine.Limit(depth=MODEL_BUILD_DEPTH))
                            total_moves += 1
                            if result.move in template:
                                hits += 1
                        hit_rate = (hits / total_moves) if total_moves > 0 else 0
                        result_queue.put(("test_elo", elo, hit_rate))

                except Exception as e:
                    logging.error(f"Error in engine_worker task: {repr(e)}")
                    # Put a failure marker in the queue if needed
                    if task_type == "analyze_position":
                        result_queue.put(("analyze_position", data, set()))
                    elif task_type == "test_elo":
                        result_queue.put(("test_elo", data[0], None))

    except Exception as e:
        logging.error(f"FATAL error starting engine in worker: {repr(e)}")

# --- UTILITY AND REPORTING FUNCTIONS ---

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

        pdf_path = PROJECT_FOLDER / f"Chess_Analysis_Report_FINAL_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(pdf_path)
        logging.info(f"Final report saved to {pdf_path}")
    except Exception as e:
        logging.error(f"Failed to generate final PDF report. Error: {repr(e)}")


def process_one_game(game, game_num):
    """
    Encapsulates the entire analysis pipeline for a single game using a
    persistent engine pool to prevent deadlocks.
    """
    # --- Setup a single pool for all of this game's analysis ---
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    processes = [multiprocessing.Process(target=engine_worker, args=(task_queue, result_queue)) for _ in range(NUM_PROCESSES)]
    for p in processes:
        p.start()

    try:
        # --- Phase 1: Generate Move Templates ---
        logging.info("Step 1: Generating move templates using engine pool...")
        board = game.board()
        fens_to_analyze = [board.fen()]
        for move in game.mainline_moves():
            board.push(move)
            fens_to_analyze.append(board.fen())
        
        for fen in fens_to_analyze:
            task_queue.put(("analyze_position", fen))

        move_templates = {}
        for _ in range(len(fens_to_analyze)):
            try:
                _, fen, template = result_queue.get(timeout=ORACLE_ANALYSIS_DEPTH * 5) # Generous timeout
                move_templates[fen] = template
            except queue.Empty:
                logging.error("Timeout waiting for analysis result from worker.")
        logging.info("Step 1: Move template generation complete.")

        generate_move_log(game_num, game, move_templates)

        # --- Phase 2: Build Performance Model ---
        logging.info("Step 2: Building performance model using engine pool...")
        player_color = chess.WHITE if game.headers.get("White") == PLAYER_TO_ANALYZE else chess.BLACK
        player_specific_templates = {fen: tmpl for fen, tmpl in move_templates.items() if chess.Board(fen).turn == player_color}
        
        elo_levels = np.arange(1350, 3200, 50)
        for elo in elo_levels:
            task_queue.put(("test_elo", (elo, player_specific_templates)))

        hit_rates_data = []
        ratings_data = []
        for _ in range(len(elo_levels)):
            try:
                _, elo, hit_rate = result_queue.get(timeout=MODEL_BUILD_DEPTH * 55) # Generous timeout
                if hit_rate is not None:
                    ratings_data.append(elo)
                    hit_rates_data.append(hit_rate)
            except queue.Empty:
                logging.error("Timeout waiting for elo test result from worker.")
        logging.info("Step 2: Model building complete.")
        
        if len(ratings_data) < 2:
            logging.warning("Not enough data to build a model for this game.")
            return None
        
        ratings = np.array(ratings_data).reshape(-1, 1)
        hit_rates = np.array(hit_rates_data)
        model = LinearRegression()
        model.fit(ratings, hit_rates)
        r_squared = model.score(ratings, hit_rates)
        
        logging.info(f"Model created for this game. R-squared: {r_squared:.4f}")

        # --- Phase 3: Final Calculations and Plotting ---
        plt.figure(figsize=(10, 6))
        plt.scatter(ratings, hit_rates, alpha=0.7, label="Engine Performance (Hit Rate)")
        plt.plot(ratings, model.predict(ratings), color='red', linewidth=2, label="Linear Regression Model")
        plt.title(f"Game {game_num}: Engine Rating vs. Hit Rate")
        plt.xlabel("Engine Elo Rating (UCI Setting)")
        plt.ylabel("Hit Rate")
        plt.grid(True); plt.legend()
        graph_path = PROJECT_FOLDER / f"performance_graph_game_{game_num}.png"
        plt.savefig(graph_path); plt.close()

        player_results = analyze_player_hits(game, move_templates)
        player_data = player_results.get(PLAYER_TO_ANALYZE, {'hits': 0, 'moves': 0})
        hit_rate = (player_data['hits'] / player_data['moves']) if player_data['moves'] > 0 else 0
        est_rating = estimate_rating_from_hit_rate(model, hit_rate)
        
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
        return game_data_for_session

    finally:
        # --- Final Phase: Cleanly shut down the engine pool ---
        for _ in range(NUM_PROCESSES):
            task_queue.put(("STOP", None))
        for p in processes:
            p.join()


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

            game_results = process_one_game(game, game_num)

            if game_results:
                session_data['completed_games_data'].append(game_results)
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data, pgn_path)
                logging.info(f"--- Finished Analysis for Game {game_num}. Results saved. ---")
            else:
                logging.error(f"Analysis for Game {game_num} failed. Skipping.")
                session_data['games_to_process_indices'].pop(0)
                save_session(session_data, pgn_path)

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
    # Set start method to 'spawn' for better stability on all platforms
    multiprocessing.set_start_method('spawn', force=True)
    main()
