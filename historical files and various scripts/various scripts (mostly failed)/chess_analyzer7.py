# chess_analyzer_real_engines.py
# A script to analyze PGN chess games and estimate player ratings.
# This version is configured to use a list of real engines from a CSV file.

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd
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

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
# This is your main analysis engine, used to generate the "best move" templates.
# Ensure this points to your compatible Stockfish executable.
ORACLE_ENGINE_PATH = PROJECT_FOLDER / "engines" / "stockfish.exe"
ANALYSIS_TIME_LIMIT = 0.5
ALT_MOVE_TOLERANCE_CP = 10

# --- SET THIS TO 'True' TO USE YOUR CSV FILE OF REAL ENGINES ---
USE_REAL_ENGINES = True

# The player name you want to track for an overall average rating.
PLAYER_TO_AVERAGE = "Desjardins373"

# --- SCRIPT SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if not ORACLE_ENGINE_PATH.is_file():
    logging.error(f"FATAL: Oracle chess engine not found at '{ORACLE_ENGINE_PATH}'")
    exit()

# --- CORE FUNCTIONS ---

def get_move_template(board, engine):
    """Analyzes a board position with the oracle engine to get the best moves."""
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

def get_real_engine_performance(move_templates):
    """
    Analyzes a list of real engines from a CSV file to get their hit rates.
    """
    logging.info("--- Building model from real engines listed in real_engines.csv ---")
    engines_csv_path = PROJECT_FOLDER / "real_engines.csv"
    if not engines_csv_path.is_file():
        logging.error(f"FATAL: real_engines.csv not found. Cannot run real engine analysis.")
        return None, None, None, None, None

    try:
        df = pd.read_csv(engines_csv_path)
        if df.empty:
            logging.error("FATAL: real_engines.csv is empty. Please add known engines to it.")
            return None, None, None, None, None
    except Exception as e:
        logging.error(f"Failed to read or parse real_engines.csv. Error: {e}")
        return None, None, None, None, None
        
    engine_paths = df['path'].tolist()
    ratings = df['rating'].values
    hit_rates_data = []

    for i, engine_path_str in enumerate(engine_paths):
        engine_path = Path(engine_path_str)
        if not engine_path.is_file():
            logging.warning(f"Skipping known engine: File not found at {engine_path}")
            hit_rates_data.append(np.nan)
            continue
        
        logging.info(f"  -> Testing with {engine_path.name} (Rating: {ratings[i]})")
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
            hit_rates_data.append(hit_rate)
        except Exception as e:
            logging.error(f"Failed to analyze with {engine_path.name}. Error: {e}")
            hit_rates_data.append(np.nan)

    hit_rates = np.array(hit_rates_data)
    valid_indices = ~np.isnan(hit_rates)
    if not np.any(valid_indices) or len(ratings[valid_indices]) < 2:
        logging.error("Could not get a valid performance from enough known engines to build a model.")
        return None, None, None, None, None
    
    valid_ratings = ratings[valid_indices].reshape(-1, 1)
    valid_hit_rates = hit_rates[valid_indices]

    model = LinearRegression()
    model.fit(valid_ratings, valid_hit_rates)
    r_squared = model.score(valid_ratings, valid_hit_rates)
    mse = mean_squared_error(valid_hit_rates, model.predict(valid_ratings))
    
    logging.info(f"Model created for this game. R-squared: {r_squared:.4f}, MSE: {mse:.4f}")
    return model, r_squared, mse, valid_ratings, valid_hit_rates

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
            if player_name not in player_hits:
                player_hits[player_name] = {'hits': 0, 'moves': 0}
            player_hits[player_name]['moves'] += 1
            if move in template_for_pos:
                player_hits[player_name]['hits'] += 1
        board.push(move)
    return player_hits

def estimate_rating_from_hit_rate(model, hit_rate):
    m, c = model.coef_[0], model.intercept_
    if m == 0: return 0
    return int((hit_rate - c) / m)

def update_running_log(log_path, game_num, game_data, running_avg_data):
    """Appends the latest game analysis and running average to the log file."""
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            if game_num == 1:
                f.write("Chess Performance Analysis - Live Log\n")
                f.write(f"Analysis started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n")

            f.write(f"\n--- Analysis for Game {game_num}: {game_data['white']} vs. {game_data['black']} ---\n")
            for player_name, results in game_data['player_results'].items():
                hit_rate = (results['hits'] / results['moves']) if results['moves'] > 0 else 0
                est_rating = estimate_rating_from_hit_rate(game_data['model'], hit_rate)
                f.write(f"  Player: {player_name}\n")
                f.write(f"    - Hit Rate: {hit_rate:.2%} ({results['hits']}/{results['moves']} moves)\n")
                f.write(f"    - Estimated Game Rating: {est_rating} Elo\n")
            
            if running_avg_data:
                f.write("\n--- Running Average Update ---\n")
                f.write(f"  Player: {running_avg_data['player_name']}\n")
                f.write(f"  Games Analyzed: {running_avg_data['game_count']}\n")
                f.write(f"  Current Average Rating: {running_avg_data['average_rating']} Elo\n")
                f.write(f"  Current 95% CI: [{running_avg_data['lower_bound']}, {running_avg_data['upper_bound']}]\n")
            f.write("="*50 + "\n")
    except Exception as e:
        logging.error(f"Could not update running log file. Error: {e}")

def generate_report_pdf(all_games_data, average_rating_data):
    """Generates a PDF report summarizing the analysis."""
    logging.info("Generating PDF report...")
    try:
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(w=0, h=10, text="Chess Performance Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(w=0, h=5, text=f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)

        for i, data in enumerate(all_games_data):
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(w=0, h=10, text=f"Analysis for Game {i+1}: {data['white']} vs. {data['black']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            if Path(data['graph_path']).is_file():
                pdf.image(data['graph_path'], w=180)
            else:
                pdf.multi_cell(w=0, h=5, text="[Graph image not found]")
            
            pdf.ln(5)

            pdf.set_font("Helvetica", '', 10)
            m, c = data['model'].coef_[0], data['model'].intercept_
            pdf.multi_cell(w=0, h=5, text=f"Model Equation: Hit Rate = {m:.6f} * Rating + {c:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.multi_cell(w=0, h=5, text=f"Model Fit (R-squared): {data['r_squared']:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            
            for player_name, results in data['player_results'].items():
                pdf.set_font("Helvetica", 'B', 11)
                pdf.cell(w=0, h=8, text=f"Player: {player_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", '', 10)
                hit_rate = (results['hits'] / results['moves']) if results['moves'] > 0 else 0
                est_rating = estimate_rating_from_hit_rate(data['model'], hit_rate)
                error_margin = 2 * np.sqrt(data['mse']) / np.abs(m)
                lower_bound, upper_bound = int(est_rating - error_margin), int(est_rating + error_margin)
                
                pdf.multi_cell(w=0, h=5, text=f"  - Hit Rate: {hit_rate:.2%} ({results['hits']}/{results['moves']} moves)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.multi_cell(w=0, h=5, text=f"  - Estimated Game Rating: {est_rating} Elo", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.multi_cell(w=0, h=5, text=f"  - 95% Confidence Interval (Approx.): [{lower_bound}, {upper_bound}]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(3)
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
            pdf.multi_cell(w=0, h=5, text=f"Combined 95% Confidence Interval: [{average_rating_data['lower_bound']}, {average_rating_data['upper_bound']}]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            pdf.multi_cell(w=0, h=5, text="Note: The combined confidence interval is calculated by pooling the variance from each individual game's model, providing a more robust estimate of overall performance.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

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
[Date "2025.07.02"]
[Round "?"]
[White "Player A"]
[Black "Desjardins373"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3
O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 12. cxb5 axb5 13. Nc3 Bb7 14. Bg5 b4 15.
Nb1 h6 16. Bh4 c5 17. dxe5 Nxe4 18. Bxe7 Qxe7 19. exd6 Qf6 20. Nbd2 Nxd6 21.
Nc4 Nxc4 22. Bxc4 Nb6 23. Ne5 Rae8 24. Bxf7+ Rxf7 25. Nxf7 Rxe1+ 26. Qxe1
Kxf7 27. Qe3 Qg5 28. Qxg5 hxg5 29. b3 Ke6 30. a3 Kd6 31. axb4 cxb4 32. Ra5
Nd5 33. f3 Bc8 34. Kf2 Bf5 35. Ra7 g6 36. Ra6+ Kc5 37. Ke1 Nf4 38. g3 Nxh3
39. Kd2 Kb5 40. Rd6 Kc5 41. Ra6 Nf2 42. g4 Bd3 43. Re6 1-0
"""
            with open(pgn_path, "w") as f:
                f.write(sample_pgn_content)
            logging.info(f"Using sample PGN file: {pgn_path}")
        else:
            logging.error(f"PGN file not found at {pgn_path}")
            return

    all_games_data = []
    player_ratings_list = []
    player_variances_list = []
    
    log_path = PROJECT_FOLDER / "running_analysis_log.txt"
    if log_path.exists():
        log_path.unlink()

    try:
        with chess.engine.SimpleEngine.popen_uci(ORACLE_ENGINE_PATH) as oracle_engine:
            with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
                game_num = 0
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None: break
                    game_num += 1
                    logging.info(f"--- Processing Game {game_num}: {game.headers.get('White', '?')} vs. {game.headers.get('Black', '?')} ---")

                    # Step 1: Generate move templates using the full-strength engine
                    move_templates = {}
                    board = game.board()
                    for move in game.mainline_moves():
                        move_templates[board.fen()] = get_move_template(board, oracle_engine)
                        board.push(move)

                    # Step 2: Build the performance model
                    if USE_REAL_ENGINES:
                        model_results = get_real_engine_performance(move_templates)
                    else:
                        # Fallback to simulation if not using real engines
                        sim_ratings, sim_hit_rates = simulate_engine_performance_data(move_templates)
                        model, r_squared, mse = generate_linear_model(sim_ratings, sim_hit_rates)
                        model_results = (model, r_squared, mse, sim_ratings, sim_hit_rates)

                    if model_results[0] is None:
                        logging.warning(f"Skipping Game {game_num} due to failure in model generation.")
                        continue
                    
                    model, r_squared, mse, ratings, hit_rates = model_results
                    
                    # Step 3: Plot the model
                    plt.figure(figsize=(10, 6))
                    plt.scatter(ratings, hit_rates, alpha=0.6, label="Engine Performance")
                    plt.plot(ratings, model.predict(ratings), color='red', linewidth=2, label="Linear Regression Model")
                    plt.title(f"Game {game_num}: Engine Rating vs. Hit Rate")
                    plt.xlabel("Engine Elo Rating")
                    plt.ylabel("Fractional Hits")
                    plt.grid(True); plt.legend()
                    graph_path = PROJECT_FOLDER / f"performance_graph_game_{game_num}.png"
                    plt.savefig(graph_path); plt.close()

                    # Step 4: Analyze human players
                    player_results = analyze_player_hits(game, move_templates)
                    
                    game_data = {
                        'white': game.headers.get('White'), 'black': game.headers.get('Black'),
                        'model': model, 'r_squared': r_squared, 'mse': mse,
                        'graph_path': str(graph_path), 'player_results': player_results
                    }
                    all_games_data.append(game_data)
                    
                    # Step 5: Update running average and log file
                    running_avg_data = None
                    if PLAYER_TO_AVERAGE in player_results:
                        player_data = player_results[PLAYER_TO_AVERAGE]
                        hit_rate = (player_data['hits'] / player_data['moves']) if player_data['moves'] > 0 else 0
                        est_rating = estimate_rating_from_hit_rate(model, hit_rate)
                        player_ratings_list.append(est_rating)
                        
                        m = model.coef_[0]
                        if m != 0:
                            se = np.sqrt(mse) / np.abs(m)
                            player_variances_list.append(se**2)

                        avg_rating = int(np.mean(player_ratings_list))
                        n = len(player_variances_list)
                        if n > 0:
                            pooled_variance = np.sum(player_variances_list) / (n**2)
                            pooled_se = np.sqrt(pooled_variance)
                            error_margin = 1.96 * pooled_se
                            
                            running_avg_data = {
                                "player_name": PLAYER_TO_AVERAGE,
                                "game_count": n,
                                "average_rating": avg_rating,
                                "lower_bound": int(avg_rating - error_margin),
                                "upper_bound": int(avg_rating + error_margin),
                            }
                    
                    update_running_log(log_path, game_num, game_data, running_avg_data)

    except Exception as e:
        logging.error(f"A critical error occurred during main processing: {e}")
        return

    if not all_games_data:
        logging.error("No games were successfully analyzed.")
        return
    
    final_average_data = running_avg_data if 'running_avg_data' in locals() else None
    
    generate_report_pdf(all_games_data, final_average_data)

if __name__ == "__main__":
    main()
