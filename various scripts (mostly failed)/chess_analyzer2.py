# chess_analyzer.py (Corrected Logic)
# A script to analyze PGN chess games, estimate player ratings, and generate a report.

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from fpdf import FPDF
from pathlib import Path
import datetime
import logging

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
ENGINE_PATH = PROJECT_FOLDER / "engines" / "stockfish.exe"
ANALYSIS_TIME_LIMIT = 0.5
ALT_MOVE_TOLERANCE_CP = 10
ORACLE_ENGINE_RATING = 3600

# SET TO 'False' to use the fast simulation.
# SET TO 'True' to use your real downloaded engines (requires real_engines.csv).
USE_REAL_ENGINES = False

# --- SCRIPT SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if not ENGINE_PATH.is_file():
    logging.error(f"FATAL: Chess engine not found at '{ENGINE_PATH}'")
    exit()

# --- CORE FUNCTIONS ---

def get_move_template(board, engine):
    try:
        info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME_LIMIT), multipv=3)
        if not info or 'pv' not in info[0]: return set()
        
        top_move_score = info[0]['score'].white().score(mate_ok=True)
        if top_move_score is None: return {info[0]['pv'][0]}

        template_moves = {info[0]['pv'][0]}
        for i in range(1, len(info)):
            if 'pv' not in info[i] or 'score' not in info[i]: continue
            move_score = info[i]['score'].white().score(mate_ok=True)
            if move_score is not None and abs(top_move_score - move_score) <= ALT_MOVE_TOLERANCE_CP:
                template_moves.add(info[i]['pv'][0])
        return template_moves
    except chess.engine.EngineTerminatedError:
        logging.error("Oracle engine terminated unexpectedly while generating a move template.")
        return set()

def get_real_engine_performance(move_templates):
    logging.info("--- Starting analysis of real engines to build model ---")
    engines_csv_path = PROJECT_FOLDER / "real_engines.csv"
    if not engines_csv_path.is_file():
        logging.error(f"FATAL: real_engines.csv not found. Cannot run real engine analysis.")
        return np.array([]), np.array([])

    df = pd.read_csv(engines_csv_path)
    engine_paths = df['path'].tolist()
    ratings = df['rating'].values.reshape(-1, 1)
    hit_rates = []

    for i, engine_path_str in enumerate(engine_paths):
        engine_path = Path(engine_path_str)
        if not engine_path.is_file():
            logging.warning(f"Skipping: Engine not found at {engine_path}")
            hit_rates.append(np.nan) # Use NaN for missing engines
            continue
        
        logging.info(f"Analyzing game with {engine_path.name} (Rating: {ratings[i][0]})")
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
            logging.info(f"Finished {engine_path.name}. Hit Rate: {hit_rate:.2%}")
        except Exception as e:
            logging.error(f"Failed to analyze with {engine_path.name}. Error: {e}")
            hit_rates.append(np.nan) # Use NaN for failed engines

    # Filter out failed engines (NaNs) before returning
    valid_indices = ~np.isnan(hit_rates)
    return ratings[valid_indices], np.array(hit_rates)[valid_indices]

def simulate_engine_performance_data(move_templates):
    logging.info("Simulating performance data for 100 engines to build model...")
    # The number of moves in the template gives us a basis for simulation
    num_positions = len(move_templates)
    ratings = np.linspace(1000, 3500, 100).reshape(-1, 1)
    
    # Simulate hit rates. A higher-rated engine is more likely to match the oracle.
    # We add noise to make it realistic.
    base_hit_prob = ratings.flatten() / 4200 # Adjusted denominator
    noise = np.random.normal(0, 0.08, ratings.shape[0])
    hit_rates = base_hit_prob + noise
    hit_rates = np.clip(hit_rates, 0, 1.0)
    return ratings, hit_rates

def generate_linear_model(ratings, hit_rates):
    model = LinearRegression()
    model.fit(ratings, hit_rates)
    r_squared = model.score(ratings, hit_rates)
    mse = mean_squared_error(hit_rates, model.predict(ratings))
    logging.info(f"Model created for this game. R-squared: {r_squared:.4f}, MSE: {mse:.4f}")
    return model, r_squared, mse

def analyze_player_hits(game, move_templates):
    player_hits = {
        game.headers.get("White", "White"): {'hits': 0, 'moves': 0},
        game.headers.get("Black", "Black"): {'hits': 0, 'moves': 0}
    }
    board = game.board()
    for move in game.mainline_moves():
        fen = board.fen()
        template_for_pos = move_templates.get(fen)
        if template_for_pos:
            player_key = "White" if board.turn == chess.WHITE else "Black"
            player_name = game.headers.get(player_key, player_key)
            player_hits[player_name]['moves'] += 1
            if move in template_for_pos:
                player_hits[player_name]['hits'] += 1
        board.push(move)
    return player_hits

def estimate_rating_from_hit_rate(model, hit_rate):
    m, c = model.coef_[0], model.intercept_
    if m == 0: return 0
    return int((hit_rate - c) / m)
    
def generate_report_pdf(all_games_data):
    logging.info("Generating PDF report...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Chess Performance Analysis Report", 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 5, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
    pdf.ln(10)

    for i, data in enumerate(all_games_data):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Analysis for Game {i+1}: {data['white']} vs. {data['black']}", 0, 1, 'L')
        pdf.set_font("Arial", '', 10)
        
        # Add the graph for this game's model
        pdf.image(data['graph_path'], x=10, y=None, w=180)
        pdf.ln(5)
        
        m, c = data['model'].coef_[0], data['model'].intercept_
        pdf.multi_cell(0, 5, f"Model Equation: Hit Rate = {m:.6f} * Rating + {c:.4f}")
        pdf.multi_cell(0, 5, f"Model Fit (R-squared): {data['r_squared']:.4f}")
        pdf.ln(5)
        
        # Player results
        for player_name, results in data['player_results'].items():
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, f"Player: {player_name}", 0, 1, 'L')
            pdf.set_font("Arial", '', 10)
            hit_rate = (results['hits'] / results['moves']) if results['moves'] > 0 else 0
            est_rating = estimate_rating_from_hit_rate(data['model'], hit_rate)
            error_margin = 2 * np.sqrt(data['mse']) / np.abs(m)
            lower_bound, upper_bound = int(est_rating - error_margin), int(est_rating + error_margin)
            
            pdf.cell(0, 5, f"  - Hit Rate: {hit_rate:.2%} ({results['hits']}/{results['moves']} moves)", 0, 1, 'L')
            pdf.cell(0, 5, f"  - Estimated Game Rating: {est_rating} Elo", 0, 1, 'L')
            pdf.cell(0, 5, f"  - 95% Confidence Interval (Approx.): [{lower_bound}, {upper_bound}]", 0, 1, 'L')
            pdf.ln(3)
        pdf.ln(5)

    pdf_path = PROJECT_FOLDER / f"Chess_Analysis_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_path)
    logging.info(f"Report saved to {pdf_path}")

def main():
    pgn_path_str = input(f"Enter the full path to your PGN file (or press Enter to use a sample): ")
    pgn_path = Path(pgn_path_str) if pgn_path_str else PROJECT_FOLDER / "sample_game.pgn"
    if not pgn_path.is_file():
        if not pgn_path_str:
            # Create a sample PGN for demonstration
            # ... (sample pgn content) ...
            logging.info(f"Using sample PGN file: {pgn_path}")
        else:
            logging.error(f"PGN file not found at {pgn_path}")
            return

    all_games_data = []
    try:
        with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as oracle_engine, open(pgn_path) as pgn_file:
            #oracle_engine.configure({"UCI_Elo": ORACLE_ENGINE_RATING})
            
            game_num = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None: break
                game_num += 1
                logging.info(f"--- Processing Game {game_num}: {game.headers.get('White', '?')} vs. {game.headers.get('Black', '?')} ---")

                # Step 1: Generate move templates for this specific game
                logging.info("Generating move templates for all positions in game...")
                move_templates = {}
                board = game.board()
                for move in game.mainline_moves():
                    move_templates[board.fen()] = get_move_template(board, oracle_engine)
                    board.push(move)

                # Step 2: Build the performance model FOR THIS GAME
                if USE_REAL_ENGINES:
                    ratings, hit_rates = get_real_engine_performance(move_templates)
                else:
                    ratings, hit_rates = simulate_engine_performance_data(move_templates)

                if len(ratings) == 0:
                    logging.warning(f"Skipping Game {game_num} due to failure in model generation.")
                    continue
                
                model, r_squared, mse = generate_linear_model(ratings, hit_rates)

                # Step 3: Plot the model for this game
                plt.figure(figsize=(10, 6))
                plt.scatter(ratings, hit_rates, alpha=0.6, label="Engine Performance")
                plt.plot(ratings, model.predict(ratings), color='red', linewidth=2, label="Linear Regression Model")
                plt.title(f"Game {game_num}: Engine Rating vs. Hit Rate")
                plt.xlabel("Engine Elo Rating")
                plt.ylabel("Fractional Hits")
                plt.grid(True); plt.legend()
                graph_path = PROJECT_FOLDER / f"performance_graph_game_{game_num}.png"
                plt.savefig(graph_path); plt.close()

                # Step 4: Analyze player hits for this game
                player_results = analyze_player_hits(game, move_templates)
                
                all_games_data.append({
                    'white': game.headers.get('White'), 'black': game.headers.get('Black'),
                    'model': model, 'r_squared': r_squared, 'mse': mse,
                    'graph_path': str(graph_path), 'player_results': player_results
                })
    except chess.engine.EngineTerminatedError:
         logging.error("The oracle engine process died. This is likely a CPU compatibility issue.")
         logging.error("Please download a different Stockfish version (e.g., 'bmi2' or 'popcnt') and try again.")
         return
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return

    if not all_games_data:
        logging.error("No games were successfully analyzed.")
        return

    # Step 5: Generate the final report for all analyzed games
    generate_report_pdf(all_games_data)

if __name__ == "__main__":
    main()