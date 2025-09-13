# engine_rater.py
# A script to estimate the Elo rating of an unknown chess engine.

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pathlib import Path
import datetime
import logging

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
# This is your main analysis engine, used to generate the "best move" templates.
ORACLE_ENGINE_PATH = PROJECT_FOLDER / "engines" / "stockfish.exe" 
ANALYSIS_TIME_LIMIT = 0.5
ALT_MOVE_TOLERANCE_CP = 10

# --- SCRIPT SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if not ORACLE_ENGINE_PATH.is_file():
    logging.error(f"FATAL: Oracle chess engine not found at '{ORACLE_ENGINE_PATH}'")
    exit()

# --- CORE FUNCTIONS (from previous script) ---

def get_move_template(board, engine):
    try:
        info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME_LIMIT), multipv=3)
        if not info or 'pv' not in info[0]: return set()
        
        top_score_obj = info[0]['score'].white()
        if top_score_obj.is_mate():
            template_moves = {info[0]['pv'][0]}
            for i in range(1, len(info)):
                if info[i].get('score') and info[i]['score'].white().is_mate():
                    template_moves.add(info[i]['pv'][0])
            return template_moves

        top_move_score = top_score_obj.score()
        template_moves = {info[0]['pv'][0]}
        for i in range(1, len(info)):
            current_score_obj = info[i].get('score')
            if current_score_obj and not current_score_obj.white().is_mate():
                move_score = current_score_obj.white().score()
                if abs(top_move_score - move_score) <= ALT_MOVE_TOLERANCE_CP:
                    template_moves.add(info[i]['pv'][0])
        return template_moves
    except Exception as e:
        logging.error(f"An unexpected error occurred in get_move_template: {e}")
        return set()

def get_known_engine_performance(move_templates):
    logging.info("--- Analyzing known engines to build baseline model ---")
    engines_csv_path = PROJECT_FOLDER / "real_engines.csv"
    if not engines_csv_path.is_file():
        logging.error(f"FATAL: real_engines.csv not found. Cannot build a baseline model.")
        return None, None

    df = pd.read_csv(engines_csv_path)
    if df.empty:
        logging.error("FATAL: real_engines.csv is empty. Please add known engines to it.")
        return None, None
        
    engine_paths = df['path'].tolist()
    ratings = df['rating'].values.reshape(-1, 1)
    hit_rates = []

    for i, engine_path_str in enumerate(engine_paths):
        engine_path = Path(engine_path_str)
        if not engine_path.is_file():
            logging.warning(f"Skipping known engine: File not found at {engine_path}")
            hit_rates.append(np.nan)
            continue
        
        logging.info(f"  -> Calibrating with {engine_path.name} (Rating: {ratings[i][0]})")
        hits, total_moves = 0, 0
        try:
            with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
                for fen, template in move_templates.items():
                    board = chess.Board(fen)
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    total_moves += 1
                    if result.move in template:
                        hits += 1
            hit_rate = (hits / total_moves) if total_moves > 0 else 0
            hit_rates.append(hit_rate)
        except Exception as e:
            logging.error(f"Failed to analyze with {engine_path.name}. Error: {e}")
            hit_rates.append(np.nan)

    valid_indices = ~np.isnan(hit_rates)
    if not np.any(valid_indices):
        logging.error("Could not get a valid performance from any known engine.")
        return None, None
        
    return ratings[valid_indices], np.array(hit_rates)[valid_indices]

def generate_linear_model(ratings, hit_rates):
    model = LinearRegression()
    model.fit(ratings, hit_rates)
    mse = mean_squared_error(hit_rates, model.predict(ratings))
    return model, mse

def estimate_rating_from_hit_rate(model, hit_rate):
    m, c = model.coef_[0], model.intercept_
    if m == 0: return 0
    return int((hit_rate - c) / m)

# --- NEW RATING FUNCTION ---
def rate_unknown_engine(engine_to_rate_path, model, move_templates, mse):
    """
    Analyzes a single unknown engine and estimates its rating.
    """
    engine_path = Path(engine_to_rate_path)
    if not engine_path.is_file():
        logging.error(f"Cannot rate engine: File not found at {engine_path}")
        return

    logging.info(f"\n--- Rating New Engine: {engine_path.name} ---")
    hits, total_moves = 0, 0
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            for fen, template in move_templates.items():
                board = chess.Board(fen)
                result = engine.play(board, chess.engine.Limit(time=0.1))
                total_moves += 1
                if result.move in template:
                    hits += 1
        
        hit_rate = (hits / total_moves) if total_moves > 0 else 0
        estimated_rating = estimate_rating_from_hit_rate(model, hit_rate)
        
        # Calculate confidence interval
        m = model.coef_[0]
        error_margin = 0
        if m != 0:
            se = np.sqrt(mse) / np.abs(m)
            error_margin = 1.96 * se

        lower_bound = int(estimated_rating - error_margin)
        upper_bound = int(estimated_rating + error_margin)

        print("\n--- RATING ESTIMATE ---")
        print(f"Engine: {engine_path.name}")
        print(f"Hit Rate: {hit_rate:.2%}")
        print(f"Estimated Elo Rating: {estimated_rating}")
        print(f"95% Confidence Interval: [{lower_bound} - {upper_bound}]")
        print("-----------------------\n")

    except Exception as e:
        logging.error(f"Failed to analyze {engine_path.name}. Error: {e}")


def main():
    """
    Main workflow to build a model and then rate an unknown engine.
    """
    # Step 1: Get a PGN file to use as the analysis benchmark
    pgn_path_str = input(f"Enter the full path to a PGN file (this will be the benchmark game): ")
    pgn_path = Path(pgn_path_str)
    if not pgn_path.is_file():
        logging.error(f"PGN file not found at {pgn_path}")
        return

    # Step 2: Generate move templates from the benchmark PGN
    move_templates = {}
    try:
        with chess.engine.SimpleEngine.popen_uci(ORACLE_ENGINE_PATH) as oracle_engine:
            with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
                # We only need one game to create the templates
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    logging.error("No valid game found in the PGN file.")
                    return
                
                logging.info(f"Generating move templates using benchmark game: {game.headers.get('White', '?')} vs. {game.headers.get('Black', '?')}")
                board = game.board()
                for move in game.mainline_moves():
                    move_templates[board.fen()] = get_move_template(board, oracle_engine)
                    board.push(move)
    except Exception as e:
        logging.error(f"A critical error occurred while creating move templates: {e}")
        return

    # Step 3: Build the baseline model using your known engines
    ratings, hit_rates = get_known_engine_performance(move_templates)
    if ratings is None or len(ratings) < 2:
        logging.error("Could not build a model. Need at least 2 known engines in real_engines.csv.")
        return

    model, mse = generate_linear_model(ratings, hit_rates)
    logging.info("Baseline performance model created successfully.")

    # Step 4: Loop to rate new engines
    while True:
        engine_to_rate_str = input(f"Enter the full path to the new engine's .exe file (or press Enter to quit): ")
        if not engine_to_rate_str:
            break
        
        rate_unknown_engine(engine_to_rate_str, model, move_templates, mse)

    print("Engine rater finished.")


if __name__ == "__main__":
    main()
