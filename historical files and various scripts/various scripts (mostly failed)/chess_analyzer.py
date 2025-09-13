# chess_analyzer.py
# A script to analyze PGN chess games, estimate player ratings, and generate a report.

import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from fpdf import FPDF
from pathlib import Path
import datetime
import logging

# --- CONFIGURATION ---

# IMPORTANT: Update this path to the full path of your Stockfish executable.
# It should be the file you downloaded and placed in the 'engines' subfolder.
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
ENGINE_PATH = PROJECT_FOLDER / "engines" / "stockfish.exe"

# Analysis time per move in seconds. Higher values are more accurate but much slower.
ANALYSIS_TIME_LIMIT = 0.5 

# Centipawn tolerance for what's considered an "equally strong" alternative move.
# 10 centipawns = 0.1 pawns.
ALT_MOVE_TOLERANCE_CP = 10

# The Elo rating of our "oracle" engine used for generating the move template.
ORACLE_ENGINE_RATING = 3600 

# --- SCRIPT SETUP ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the engine executable exists before starting
if not ENGINE_PATH.is_file():
    logging.error(f"FATAL: Chess engine not found at '{ENGINE_PATH}'")
    logging.error("Please download Stockfish, place it in the 'engines' subfolder, and update the ENGINE_PATH variable.")
    exit()

# --- CORE FUNCTIONS ---

def simulate_engine_performance_data():
    """
    Simulates the performance of 100 engines to create a baseline for our linear model.
    This avoids the need to download and run 100 separate engine executables.
    
    Returns:
        A tuple of (numpy.array, numpy.array): (ratings, hit_rates)
    """
    logging.info("Simulating performance data for 100 engines...")
    # Generate 100 engine ratings evenly distributed from 1000 to 3500
    ratings = np.linspace(1000, 3500, 100).reshape(-1, 1)
    
    # Create a plausible "hit rate" for each engine.
    # A perfect hit rate is 1.0. We assume a strong correlation with rating,
    # but add some random noise to make it realistic.
    # The base hit rate is assumed to be (rating / 4000).
    base_hit_rate = ratings.flatten() / 4000
    noise = np.random.normal(0, 0.05, ratings.shape[0])
    hit_rates = base_hit_rate + noise
    hit_rates = np.clip(hit_rates, 0, 1.0) # Ensure hit rates are between 0 and 1
    
    return ratings, hit_rates

def generate_linear_model(ratings, hit_rates):
    """
    Creates a linear regression model from the engine performance data.
    
    Args:
        ratings (numpy.array): The independent variable (engine Elo ratings).
        hit_rates (numpy.array): The dependent variable (fraction of correct moves).
        
    Returns:
        A tuple containing: (LinearRegression model, r_squared, mse)
    """
    logging.info("Generating linear regression model...")
    model = LinearRegression()
    model.fit(ratings, hit_rates)
    
    # Performance metrics
    predictions = model.predict(ratings)
    r_squared = model.score(ratings, hit_rates)
    mse = mean_squared_error(hit_rates, predictions)
    
    logging.info(f"Model created. R-squared: {r_squared:.4f}, MSE: {mse:.4f}")
    return model, r_squared, mse

def get_move_template(board, engine):
    """
    Analyzes a board position with the oracle engine to get the best move and alternatives.
    
    Args:
        board (chess.Board): The current chess position.
        engine (chess.engine.SimpleEngine): The analysis engine.
        
    Returns:
        A set of chess.Move objects considered "correct" for the position.
    """
    try:
        # Use multipv to get the top 3 moves
        info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME_LIMIT), multipv=3)
        
        if not info:
            return set()
            
        top_move_score = info[0]['score'].white().score(mate_ok=True)
        if top_move_score is None: return {info[0]['pv'][0]}

        template_moves = {info[0]['pv'][0]}
        
        # Check other moves against the tolerance
        for i in range(1, len(info)):
            move_score = info[i]['score'].white().score(mate_ok=True)
            if move_score is not None and abs(top_move_score - move_score) <= ALT_MOVE_TOLERANCE_CP:
                template_moves.add(info[i]['pv'][0])
                
        return template_moves
    except chess.engine.EngineTerminatedError:
        logging.error("Engine terminated unexpectedly. Cannot generate move template.")
        return set()


def analyze_game(game, engine, move_templates):
    """
    Calculates the hit rate for both players in a single game.
    
    Args:
        game (chess.pgn.Game): The game to analyze.
        engine (chess.engine.SimpleEngine): Not used in this function directly but kept for consistency.
        move_templates (dict): A dictionary mapping board FENs to template moves.
        
    Returns:
        A dictionary with player names as keys and their hit rate info as values.
    """
    player_hits = {
        game.headers["White"]: {'hits': 0, 'moves': 0},
        game.headers["Black"]: {'hits': 0, 'moves': 0}
    }

    board = game.board()
    for move in game.mainline_moves():
        fen = board.fen()
        template_for_pos = move_templates.get(fen)

        if template_for_pos:
            player = game.headers["White"] if board.turn == chess.WHITE else game.headers["Black"]
            player_hits[player]['moves'] += 1
            if move in template_for_pos:
                player_hits[player]['hits'] += 1
        
        board.push(move)

    results = {}
    for player, data in player_hits.items():
        hit_rate = (data['hits'] / data['moves']) if data['moves'] > 0 else 0
        results[player] = {'hit_rate': hit_rate, 'total_moves': data['moves']}
    
    return results


def estimate_rating_from_hit_rate(model, hit_rate):
    """
    Uses the linear model to estimate an Elo rating from a hit rate.
    
    Args:
        model (LinearRegression): The trained linear model.
        hit_rate (float): The player's hit rate (0 to 1.0).
        
    Returns:
        The estimated Elo rating as an integer.
    """
    # The model predicts hit_rate from rating: hit_rate = m*rating + c
    # We need to solve for rating: rating = (hit_rate - c) / m
    m = model.coef_[0]
    c = model.intercept_
    
    if m == 0: return 0 # Avoid division by zero
    
    estimated_rating = (hit_rate - c) / m
    return int(estimated_rating)
    
def generate_report_pdf(report_data):
    """
    Generates a PDF report summarizing the analysis.
    
    Args:
        report_data (dict): A dictionary containing all data needed for the report.
    """
    logging.info("Generating PDF report...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    # Title
    pdf.cell(0, 10, "Chess Performance Analysis Report", 0, 1, 'C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 5, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
    pdf.ln(10)
    
    # Model Info Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Engine Performance Model", 0, 1, 'L')
    pdf.set_font("Arial", '', 10)
    pdf.image(report_data['graph_path'], x=10, y=None, w=180)
    pdf.ln(5)
    
    m = report_data['model'].coef_[0]
    c = report_data['model'].intercept_
    pdf.multi_cell(0, 5, f"The analysis is based on a linear model derived from the simulated performance of 100 chess engines with ratings from 1000 to 3500.")
    pdf.ln(2)
    pdf.multi_cell(0, 5, f"Model Equation: Hit Rate = {m:.6f} * Rating + {c:.4f}")
    pdf.multi_cell(0, 5, f"Model Fit (R-squared): {report_data['r_squared']:.4f}")
    pdf.multi_cell(0, 5, f"Mean Squared Error: {report_data['mse']:.4f}")
    pdf.ln(10)
    
    # Game Analysis Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Player Performance Analysis", 0, 1, 'L')
    
    for player, data in report_data['player_ratings'].items():
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 8, f"Player: {player}", 0, 1, 'L')
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 5, f"  - Average Hit Rate across {data['game_count']} game(s): {data['avg_hit_rate']:.2%}", 0, 1, 'L')
        pdf.cell(0, 5, f"  - Estimated Performance Rating: {data['estimated_rating']} Elo", 0, 1, 'L')
        # Note: A true confidence interval requires more complex statistical analysis
        # of the model's prediction variance, which is approximated here.
        error_margin = 2 * np.sqrt(report_data['mse']) / np.abs(m)
        lower_bound = int(data['estimated_rating'] - error_margin)
        upper_bound = int(data['estimated_rating'] + error_margin)
        pdf.cell(0, 5, f"  - 95% Confidence Interval (Approx.): [{lower_bound}, {upper_bound}]", 0, 1, 'L')
        pdf.ln(5)
        
    # Save the PDF
    pdf_path = PROJECT_FOLDER / f"Chess_Analysis_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_path)
    logging.info(f"Report saved to {pdf_path}")


def main():
    """Main function to run the full analysis pipeline."""
    
    # --- Step 1: Build the Performance Model ---
    ratings, hit_rates = simulate_engine_performance_data()
    model, r_squared, mse = generate_linear_model(ratings, hit_rates)

    # Generate and save the performance graph
    plt.figure(figsize=(10, 6))
    plt.scatter(ratings, hit_rates, alpha=0.6, label="Simulated Engine Performance")
    plt.plot(ratings, model.predict(ratings), color='red', linewidth=2, label="Linear Regression Model")
    plt.title("Engine Performance: Rating vs. 'Correct Move' Hit Rate")
    plt.xlabel("Engine Elo Rating (Independent Variable)")
    plt.ylabel("Fractional Hits (Dependent Variable)")
    plt.grid(True)
    plt.legend()
    graph_path = PROJECT_FOLDER / "performance_graph.png"
    plt.savefig(graph_path)
    logging.info(f"Performance graph saved to {graph_path}")
    
    # --- Step 2: Analyze PGN file ---
    pgn_path_str = input(f"Enter the full path to your PGN file (or press Enter to use a sample): ")
    if not pgn_path_str:
        # Create a sample PGN for demonstration if none is provided
        sample_pgn_content = """
[Event "Sample Game"]
[Site "?"]
[Date "????.??.??"]
[Round "?"]
[White "Player A"]
[Black "Player B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3
O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15.
Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21.
Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1
Kxf7 27. Qe3 Qg5 28. Qxg5 hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5
Nd5 33. f3 Bc8 34. Kf2 Bf5 35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3
39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6 Nf2 42. g4 Bd3 43. Re6 1-0
"""
        pgn_path = PROJECT_FOLDER / "sample_game.pgn"
        with open(pgn_path, "w") as f:
            f.write(sample_pgn_content)
        logging.info(f"Using sample PGN file: {pgn_path}")
    else:
        pgn_path = Path(pgn_path_str)

    if not pgn_path.is_file():
        logging.error(f"PGN file not found at {pgn_path}")
        return

    # --- Step 3: Generate Move Templates for all positions in the PGN ---
    all_player_results = {}
    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        engine.configure({"UCI_Elo": ORACLE_ENGINE_RATING}) # Set engine strength
        
        with open(pgn_path) as pgn_file:
            game_count = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                game_count += 1
                logging.info(f"--- Analyzing Game {game_count}: {game.headers.get('White', '?')} vs {game.headers.get('Black', '?')} ---")

                # Generate templates for this game
                logging.info("Generating move templates for all positions in game...")
                move_templates = {}
                board = game.board()
                for move in game.mainline_moves():
                    fen = board.fen()
                    if fen not in move_templates:
                        move_templates[fen] = get_move_template(board, engine)
                    board.push(move)
                
                # Analyze player performance for this game
                game_results = analyze_game(game, engine, move_templates)
                
                # Store results
                for player, data in game_results.items():
                    if player not in all_player_results:
                        all_player_results[player] = {'total_hit_rate': 0.0, 'game_count': 0, 'total_moves': 0}
                    all_player_results[player]['total_hit_rate'] += data['hit_rate']
                    all_player_results[player]['game_count'] += 1
                    all_player_results[player]['total_moves'] += data['total_moves']
    
    if not all_player_results:
        logging.error("No games were analyzed. Exiting.")
        return

    # --- Step 4: Calculate Final Ratings and Generate Report ---
    final_player_ratings = {}
    for player, data in all_player_results.items():
        avg_hit_rate = (data['total_hit_rate'] / data['game_count']) if data['game_count'] > 0 else 0
        estimated_rating = estimate_rating_from_hit_rate(model, avg_hit_rate)
        final_player_ratings[player] = {
            'avg_hit_rate': avg_hit_rate,
            'estimated_rating': estimated_rating,
            'game_count': data['game_count']
        }

    report_data = {
        'model': model,
        'r_squared': r_squared,
        'mse': mse,
        'graph_path': str(graph_path),
        'player_ratings': final_player_ratings,
    }
    
    generate_report_pdf(report_data)

if __name__ == "__main__":
    main()