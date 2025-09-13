# chess_analyzer_final_uci.py
# A script to analyze PGN chess games and estimate a specific player's rating.
# This version is OPTIMIZED to run faster by using multiprocessing to analyze
# games in parallel and provides both a live-updating summary log and detailed
# move-by-move logs for each game.

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
import os

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
# This should be the path to your compatible Stockfish executable.
ENGINE_PATH = PROJECT_FOLDER / "engines" / "stockfish.exe"
ANALYSIS_TIME_LIMIT = 5.0  # Time for oracle engine to find best moves.
ALT_MOVE_TOLERANCE_CP = 10 # Centipawn tolerance for alternative moves.
MODEL_BUILD_TIME_LIMIT = 5.0 # Time for UCI elo engine to play moves.

# The player name you want to track for an overall average rating.
PLAYER_TO_ANALYZE = "Desjardins373"
# Number of CPU cores to use for analysis.
NUM_PROCESSES = 2

# --- SCRIPT SETUP ---
# The logging format is updated to show which process is logging the message.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
if not ENGINE_PATH.is_file():
    logging.error(f"FATAL: Chess engine not found at '{ENGINE_PATH}'")
    exit()

# --- CORE FUNCTIONS ---

def get_move_template(board, engine):
    """Analyzes a board position with the oracle engine to get the best moves."""
    try:
        # The oracle engine gets a longer time to find the "true" best moves.
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

def build_model_with_uci_levels(engine, move_templates_for_model):
    """
    Generates performance data by running a single engine at various Elo levels.
    """
    logging.info(f"Building performance model using {len(move_templates_for_model)} positions...")
    
    elo_levels = np.arange(1350, 3150, 50)
    ratings_data = []
    hit_rates_data = []

    for elo in elo_levels:
        try:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            
            hits, total_moves = 0, 0
            # Cap the number of positions used to build the model for speed.
            for i, (fen, template) in enumerate(move_templates_for_model.items()):
                if i >= 50: break 
                board = chess.Board(fen)
                # The Elo-limited engine gets a very short time, simulating quick decisions.
                result = engine.play(board, chess.engine.Limit(time=MODEL_BUILD_TIME_LIMIT))
                total_moves += 1
                if result.move in template:
                    hits += 1
            
            hit_rate = (hits / total_moves) if total_moves > 0 else 0
            ratings_data.append(elo)
            hit_rates_data.append(hit_rate)

        except Exception as e:
            logging.error(f"Failed to test at Elo {elo}. Error: {e}")
            continue

    engine.configure({"UCI_LimitStrength": False})

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

def log_move_by_move_analysis(game_num, game, move_templates):
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
                template_str = ", ".join(board.san(m) for m in template)
                
                played_move_san = board.san(move)
                is_hit = move in template
                hit_str = "HIT" if is_hit else "MISS"

                f.write(f"Move: {move_num_str} {played_move_san} ({player_name})\n")
                f.write(f"  - Status: {hit_str}\n")
                f.write(f"  - Template Moves: [{template_str}]\n\n")

                board.push(move)
        logging.info(f"Successfully created move-by-move log for Game {game_num}.")
    except Exception as e:
        logging.error(f"Could not create move-by-move log for Game {game_num}. Error: {e}")


def process_game(game_tuple):
    """
    Worker function to process a single chess game. This function is run in a
    separate process by the multiprocessing pool.
    """
    game, game_num = game_tuple  # Unpack arguments

    # Each process must create its own engine instance.
    try:
        with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
            logging.info(f"Processing Game {game_num}: {game.headers.get('White', '?')} vs. {game.headers.get('Black', '?')}")

            # Step 1: Generate move templates for ALL positions in the game.
            move_templates = {}
            board = game.board()
            for move in game.mainline_moves():
                move_templates[board.fen()] = get_move_template(board, engine)
                board.push(move)

            # NEW Step 2: Create the detailed move-by-move log file for this game.
            log_move_by_move_analysis(game_num, game, move_templates)

            # Step 3: Filter templates for only the target player's positions.
            player_color = chess.WHITE if game.headers.get("White") == PLAYER_TO_ANALYZE else chess.BLACK
            player_specific_templates = {}
            board = game.board()
            for move in game.mainline_moves():
                if board.turn == player_color:
                    player_specific_templates[board.fen()] = move_templates.get(board.fen(), set())
                board.push(move)

            # Step 4: Build the performance model using the player-specific positions.
            model_results = build_model_with_uci_levels(engine, player_specific_templates)
            if model_results[0] is None:
                logging.warning(f"Skipping Game {game_num} due to failure in model generation.")
                return None

            model, r_squared, mse, ratings, hit_rates = model_results
            
            # Step 5: Plot the model for this game.
            plt.figure(figsize=(10, 6))
            plt.scatter(ratings, hit_rates, alpha=0.6, label="Engine Performance at UCI Levels")
            plt.plot(ratings, model.predict(ratings), color='red', linewidth=2, label="Linear Regression Model")
            plt.title(f"Game {game_num}: Engine Rating vs. Hit Rate ({game.headers.get('White')} vs. {game.headers.get('Black')})")
            plt.xlabel("Engine Elo Rating (UCI Setting)")
            plt.ylabel("Fractional Hits")
            plt.grid(True); plt.legend()
            graph_path = PROJECT_FOLDER / f"performance_graph_game_{game_num}.png"
            plt.savefig(graph_path); plt.close()

            # Step 6: Analyze human players' hit rates against the full template.
            player_results = analyze_player_hits(game, move_templates)
            
            # Step 7: Package results to be returned to the main process.
            game_data = {
                'game_num': game_num,
                'white': game.headers.get('White'), 'black': game.headers.get('Black'),
                'model': model, 'r_squared': r_squared, 'mse': mse,
                'graph_path': str(graph_path), 'player_results': player_results
            }
            logging.info(f"Finished processing Game {game_num}.")
            return game_data

    except Exception as e:
        logging.error(f"A critical error occurred in worker for game {game_num}: {e}")
        return None


def update_running_log(log_path, game_num, game_data, running_avg_data):
    """Appends the latest game analysis and running average to the log file."""
    try:
        # Write the header only if the file is new (i.e., for the first game processed)
        write_header = not log_path.exists()
        with open(log_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("Chess Performance Analysis - Live Log\n")
                f.write(f"Analysis started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Target Player: {PLAYER_TO_ANALYZE}\n")
                f.write("="*50 + "\n")

            f.write(f"\n--- Analysis for Game {game_num}: {game_data['white']} vs. {game_data['black']} ---\n")
            
            if PLAYER_TO_ANALYZE in game_data['player_results']:
                results = game_data['player_results'][PLAYER_TO_ANALYZE]
                hit_rate = (results['hits'] / results['moves']) if results['moves'] > 0 else 0
                est_rating = estimate_rating_from_hit_rate(game_data['model'], hit_rate)
                f.write(f"  Player: {PLAYER_TO_ANALYZE}\n")
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
    logging.info("Generating final PDF report...")
    try:
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(w=0, h=10, text="Chess Performance Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(w=0, h=5, text=f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)

        # IMPORTANT: Sort the data by game number before generating the report
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
            m, c = data['model'].coef_[0], data['model'].intercept_
            pdf.multi_cell(w=0, h=5, text=f"Model Equation: Hit Rate = {m:.6f} * Rating + {c:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.multi_cell(w=0, h=5, text=f"Model Fit (R-squared): {data['r_squared']:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            
            if PLAYER_TO_ANALYZE in data['player_results']:
                results = data['player_results'][PLAYER_TO_ANALYZE]
                pdf.set_font("Helvetica", 'B', 11)
                pdf.cell(w=0, h=8, text=f"Player: {PLAYER_TO_ANALYZE}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", '', 10)
                hit_rate = (results['hits'] / results['moves']) if results['moves'] > 0 else 0
                est_rating = estimate_rating_from_hit_rate(data['model'], hit_rate)
                error_margin = 2 * np.sqrt(data['mse']) / np.abs(m) if m != 0 else float('inf')
                lower_bound, upper_bound = int(est_rating - error_margin), int(est_rating + error_margin)
                
                pdf.multi_cell(w=0, h=5, text=f"  - Hit Rate: {hit_rate:.2%} ({results['hits']}/{results['moves']} moves)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.multi_cell(w=0, h=5, text=f"  - Estimated Game Rating: {est_rating} Elo", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.multi_cell(w=0, h=5, text=f"  - 95% Confidence Interval (Approx.): [{lower_bound}, {upper_bound}]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
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

    # Step 1: Read all games relevant to the player into a list first.
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

    # Step 2: Set up for live processing.
    all_games_data = []
    player_ratings_list = []
    player_variances_list = []
    final_average_data = None
    log_path = PROJECT_FOLDER / "running_analysis_log.txt"
    if log_path.exists():
        log_path.unlink() # Clear old log at the start.

    # Step 3: Use a multiprocessing Pool to distribute work and get live results.
    try:
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            # Use imap_unordered to get results as they complete, not all at the end.
            results_iterator = pool.imap_unordered(process_game, games_to_process)

            # Process each result as it comes in from the worker processes.
            for game_data in results_iterator:
                if game_data is None: # Skip any games that failed in the worker.
                    continue
                
                # Add the completed game's data to our master list for the final report.
                all_games_data.append(game_data)

                # --- Calculate running average and update log file immediately ---
                running_avg_data = None
                if PLAYER_TO_ANALYZE in game_data['player_results']:
                    player_data = game_data['player_results'][PLAYER_TO_ANALYZE]
                    if player_data['moves'] > 0:
                        hit_rate = player_data['hits'] / player_data['moves']
                        est_rating = estimate_rating_from_hit_rate(game_data['model'], hit_rate)
                        player_ratings_list.append(est_rating)
                        
                        m = game_data['model'].coef_[0]
                        if m != 0:
                            se = np.sqrt(game_data['mse']) / np.abs(m)
                            player_variances_list.append(se**2)

                        avg_rating = int(np.mean(player_ratings_list))
                        n = len(player_variances_list)
                        if n > 0:
                            pooled_variance = np.sum(player_variances_list) / (n**2)
                            pooled_se = np.sqrt(pooled_variance)
                            error_margin = 1.96 * pooled_se
                            
                            running_avg_data = {
                                "player_name": PLAYER_TO_ANALYZE, "game_count": n,
                                "average_rating": avg_rating,
                                "lower_bound": int(avg_rating - error_margin),
                                "upper_bound": int(avg_rating + error_margin),
                            }
                
                # Update the log file with the data from the game that just finished.
                update_running_log(log_path, game_data['game_num'], game_data, running_avg_data)
                logging.info(f"Live log updated for Game {game_data['game_num']}.")
                final_average_data = running_avg_data  # Keep the last calculated average for the PDF.

    except Exception as e:
        logging.error(f"A critical error occurred during the multiprocessing phase: {e}")
        return

    if not all_games_data:
        logging.error("No games were successfully analyzed after parallel processing.")
        return
    
    # Step 4: Generate the final PDF report after all games are done.
    generate_report_pdf(all_games_data, final_average_data)
    logging.info("Analysis complete.")

if __name__ == "__main__":
    # This is crucial for multiprocessing to work correctly on all platforms.
    multiprocessing.freeze_support()
    main()
