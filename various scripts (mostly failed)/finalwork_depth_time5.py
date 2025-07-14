# chess_analyzer_multi_engine_v35_adaptive_r2.py
# A script to analyze PGN chess games and estimate a specific player's rating.
#
# KEY CHANGE (from v34):
#   1. ADAPTIVE ANALYSIS: Implements a check on the model's R-squared value.
#      If R² is below a threshold (0.75), the script automatically re-runs the
#      analysis for that game using a time-based limit instead of depth.
#   2. BEST MODEL SELECTION: It compares the R² from the initial (depth-based)
#      and secondary (time-based) analyses, and selects the dataset that
#      produced the higher R² value to use for the final rating estimation.
#   3. ENHANCED LOGGING: The script now logs which analysis strategy was used
#      for each game and why, providing a clear audit trail in the session
#      file and the final report.

import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg') # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import logging
import multiprocessing
import json
import sys
import csv
from collections import defaultdict

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(__file__).resolve().parent
ENGINES_CSV_PATH = PROJECT_FOLDER / "real_engines.csv"
SESSION_FOLDER = PROJECT_FOLDER / "chess_analysis_session"
SESSION_FOLDER.mkdir(exist_ok=True)

PLAYER_TO_ANALYZE = "Desjardins373"
NUM_ANALYSIS_CORES = 2
NUM_ORACLE_ENGINES = 1
R_SQUARED_THRESHOLD = 0.75

# --- CCRL-INSPIRED ENGINE SETTINGS ---
HASH_SIZE_MB = 256
TABLEBASE_PATH = r"" # Set to your Syzygy path, or leave blank to disable

# --- HIT-BASED ANALYSIS CONTROLS ---
TEMPLATE_MOVE_COUNT = 3

# --- PRIMARY (DEPTH-PRIORITY) ANALYSIS CONTROLS ---
ORACLE_ANALYSIS_DEPTH = 20
ORACLE_TIMEOUT = 600 # (in seconds)

PRIMARY_MODEL_BUILD_DEPTH = 12
PRIMARY_MODEL_BUILD_TIMEOUT = 6 # (in seconds)

PRIMARY_PLAYER_ANALYSIS_DEPTH = 12
PRIMARY_PLAYER_ANALYSIS_TIMEOUT = 6 # (in seconds)

# --- SECONDARY (TIME-PRIORITY) ANALYSIS CONTROLS ---
SECONDARY_MODEL_BUILD_DEPTH = None # Use None to signify time-only analysis
SECONDARY_MODEL_BUILD_TIMEOUT = 6 # (in seconds)

SECONDARY_PLAYER_ANALYSIS_DEPTH = None # Use None to signify time-only analysis
SECONDARY_PLAYER_ANALYSIS_TIMEOUT = 6 # (in seconds)


# --- SCRIPT SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(SESSION_FOLDER / 'analysis_log.txt'), logging.StreamHandler(sys.stdout)]
)
# ... (logging setup remains the same)

# --- HELPER FUNCTIONS ---

def get_standard_engine_options(csv_options_str):
    """Parses engine options from CSV and adds standard configurations."""
    options = {"Hash": HASH_SIZE_MB}
    tb_path = Path(TABLEBASE_PATH)
    if tb_path.is_dir() and tb_path.exists():
        options["SyzygyPath"] = str(tb_path)
    if csv_options_str:
        try:
            options.update(json.loads(csv_options_str))
        except json.JSONDecodeError:
            logging.warning(f"Could not decode JSON options from CSV: {csv_options_str}")
    return options

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """Displays a terminal progress bar."""
    if total == 0: total = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: sys.stdout.write('\n'); sys.stdout.flush()

def calculate_move_quality(centipawn_loss):
    """Calculates a quality score (1.0 to 0.0) based on CPL from the best move."""
    if not isinstance(centipawn_loss, (int, float)) or centipawn_loss is None: return 0.0
    if centipawn_loss <= 0: return 1.0
    if centipawn_loss >= 100: return 0.0
    return 1.0 - (centipawn_loss / 100.0)

def calculate_punishment_score(centipawn_loss):
    """Calculates a score for a 'miss' based on CPL."""
    if not isinstance(centipawn_loss, (int, float)) or centipawn_loss is None: return 0.0
    if centipawn_loss <= 0: return 0.90
    if centipawn_loss >= 200: return 0.0
    return 0.90 * (1 - (centipawn_loss / 200.0))

# --- ISOLATED WORKER FUNCTIONS ---

def universal_worker(args):
    """A universal worker for all analysis types: oracle, model building, and player analysis."""
    (fen, player_move, engine_info, limit_params, ground_truth_template, worker_mode) = args
    # worker_mode can be 'oracle', 'model', 'player'

    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            options = get_standard_engine_options(engine_info.get('uci_options', '{}'))
            engine.configure(options)
            board = chess.Board(fen)

            # --- ORACLE MODE ---
            if worker_mode == 'oracle':
                info = engine.analyse(board, chess.engine.Limit(**limit_params), multipv=TEMPLATE_MOVE_COUNT)
                template = []
                if not info: return None

                # FIX: Get score from the perspective of the player to move for correct CPL calculation.
                pov_color = board.turn
                best_pov_score = info[0]['score'].pov(pov_color).score(mate_score=30000)
                if best_pov_score is None: return None

                for move_info in info:
                    if 'pv' in move_info and 'score' in move_info:
                        move = move_info['pv'][0]
                        current_pov_score = move_info['score'].pov(pov_color).score(mate_score=30000)
                        if current_pov_score is None: continue

                        # CPL is the difference from the best move, from the current player's perspective.
                        # A higher POV score is always better.
                        cpl = max(0, best_pov_score - current_pov_score)
                        quality_score = calculate_move_quality(cpl)
                        
                        # Get the raw score from White's perspective for the punishment calculation.
                        raw_score_white = move_info['score'].white().score(mate_score=30000)
                        if raw_score_white is None: continue

                        template.append({'move': move.uci(), 'score': quality_score, 'raw_score': raw_score_white})
                return template

            # --- MODEL OR PLAYER MODE ---
            move_to_evaluate = player_move
            if worker_mode == 'model':
                result = engine.play(board, chess.engine.Limit(**limit_params))
                if not result.move: return None
                move_to_evaluate = result.move

            # Check for a "hit" in the ground truth template
            for template_entry in ground_truth_template[fen]:
                if move_to_evaluate.uci() == template_entry['move']:
                    return template_entry['score']

            # If it's a "miss", calculate punishment score
            if not ground_truth_template[fen]:
                logging.warning(f"Ground truth template for FEN {fen} is empty. Cannot calculate punishment score.")
                return None
            best_move_score_white_pov = ground_truth_template[fen][0]['raw_score']
            
            try:
                info = engine.analyse(board, chess.engine.Limit(**limit_params), root_moves=[move_to_evaluate])
                if not info: return None
                played_move_score_white_pov = info['score'].white().score(mate_score=30000)
                if played_move_score_white_pov is None: return None

                # FIX: CPL calculation must account for whose turn it is.
                if board.turn == chess.WHITE:
                    cpl = best_move_score_white_pov - played_move_score_white_pov
                else: # Black's turn
                    cpl = played_move_score_white_pov - best_move_score_white_pov
                
                cpl = max(0, cpl) # Ensure CPL is not negative
                return calculate_punishment_score(cpl)

            except chess.engine.EngineError as ee:
                logging.warning(f"Could not analyze specific missed move {move_to_evaluate.uci()} on FEN {fen}: {ee}")
                return None

    except Exception as e:
        logging.error(f"Worker in mode '{worker_mode}' for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
    return None


# --- MAIN ANALYSIS FUNCTIONS ---

def get_ground_truth_template(game, oracle_engines, session_data, pgn_path):
    """Generates or loads the 'answer key' of best moves for the game."""
    in_progress_data = session_data.get('in_progress_game', {})
    if in_progress_data.get('offset') == game.offset and 'ground_truth_template' in in_progress_data:
        logging.info("Loading pre-calculated ground truth template from session file.")
        return in_progress_data['ground_truth_template']

    logging.info("Generating ground truth template...")
    player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    fens_to_analyze = [node.board().fen() for node in game.mainline() if node.board().turn == player_color]
    if not fens_to_analyze: return {}

    template = {}
    limit = {'depth': ORACLE_ANALYSIS_DEPTH, 'time': ORACLE_TIMEOUT}
    for engine_info in oracle_engines:
        tasks = [(fen, None, engine_info, limit, None, 'oracle') for fen in fens_to_analyze]
        print_progress_bar(0, len(tasks), prefix=f'Truth Template ({engine_info["name"]}):', suffix='Complete', length=50)
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            for i, result in enumerate(pool.imap_unordered(universal_worker, tasks)):
                if result:
                    fen_for_result = tasks[i][0]
                    template[fen_for_result] = result
                print_progress_bar(i + 1, len(tasks), prefix=f'Truth Template ({engine_info["name"]}):', suffix='Complete', length=50)

    session_data['in_progress_game']['ground_truth_template'] = template
    save_session(session_data, pgn_path)
    return template

def build_model(ground_truth_template, model_engines, session_data, pgn_path, analysis_params):
    """Builds a regression model mapping engine Elo to quality scores using specified analysis parameters."""
    logging.info(f"Building performance model with params: {analysis_params['name']}...")
    # Use a unique key for storing results based on the analysis parameter name
    results_key = f"model_engine_results_{analysis_params['name']}"
    engine_scores = session_data.get('in_progress_game', {}).get(results_key, {})
    fens_for_model = list(ground_truth_template.keys())

    limit = {k: v for k, v in analysis_params.items() if k in ['depth', 'time'] and v is not None}

    for i, engine_info in enumerate(model_engines):
        if engine_info['name'] in engine_scores:
            logging.info(f"Skipping model engine {engine_info['name']} for '{analysis_params['name']}' (already processed).")
            continue
        logging.info(f"Processing model engine {i+1}/{len(model_engines)}: {engine_info['name']}")
        tasks = [(fen, None, engine_info, limit, ground_truth_template, 'model') for fen in fens_for_model]
        scores_list = []
        print_progress_bar(0, len(tasks), prefix=f'Building ({engine_info["name"]} @ {analysis_params["name"]}):', suffix='Complete', length=50)
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            for j, score in enumerate(pool.imap_unordered(universal_worker, tasks)):
                if score is not None: scores_list.append(score)
                print_progress_bar(j + 1, len(tasks), prefix=f'Building ({engine_info["name"]} @ {analysis_params["name"]}):', suffix='Complete', length=50)
        if scores_list:
            engine_scores[engine_info['name']] = np.mean(scores_list)
            session_data['in_progress_game'][results_key] = engine_scores
            save_session(session_data, pgn_path)

    ratings_data, quality_scores_data = [], []
    for eng_info in model_engines:
        if eng_info['name'] in engine_scores:
            ratings_data.append(eng_info['rating'])
            quality_scores_data.append(engine_scores[eng_info['name']])

    if len(ratings_data) < 3:
        logging.warning("Not enough data points (<3) to build a reliable model.")
        return None
    
    ratings = np.array(ratings_data).reshape(-1, 1)
    quality_scores = np.array(quality_scores_data)
    
    # Fit both linear and polynomial models
    linear_model = LinearRegression().fit(ratings, quality_scores)
    poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(ratings, quality_scores)
    
    # Check if the polynomial model is monotonically increasing
    test_ratings = np.arange(ratings.min(), ratings.max(), 10).reshape(-1, 1)
    poly_predictions = poly_model.predict(test_ratings)
    is_monotonic = np.all(np.diff(poly_predictions) >= 0)

    linear_r2 = r2_score(quality_scores, linear_model.predict(ratings))
    poly_r2 = r2_score(quality_scores, poly_model.predict(ratings)) if is_monotonic else -1.0 # Penalize non-monotonic

    # Choose the best model
    if poly_r2 > linear_r2:
        best_model, best_r2, best_model_name = poly_model, poly_r2, 'Polynomial'
    else:
        best_model, best_r2, best_model_name = linear_model, linear_r2, 'Linear'

    logging.info(f"Model selection for '{analysis_params['name']}': Best is {best_model_name}, R²={best_r2:.4f}")

    return {
        'model': best_model, 'r_squared': best_r2, 'ratings': ratings,
        'quality_scores': quality_scores, 'model_type': best_model_name,
        'raw_model_results': engine_scores
    }


def analyze_player(game, ground_truth_template, analysis_engine_info, analysis_params):
    """Analyzes the player's moves using specified analysis parameters."""
    logging.info(f"Analyzing player moves with params: {analysis_params['name']}...")
    player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    tasks = []
    board = game.board()
    limit = {k: v for k, v in analysis_params.items() if k in ['depth', 'time'] and v is not None}

    for move in game.mainline_moves():
        if board.turn == player_color:
            fen = board.fen()
            if fen in ground_truth_template:
                tasks.append((fen, move, analysis_engine_info, limit, ground_truth_template, 'player'))
        board.push(move)
    if not tasks: return 0.0, 0

    scores = []
    print_progress_bar(0, len(tasks), prefix=f'Player Analysis ({analysis_params["name"]}):', suffix='Complete', length=50)
    with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
        for i, score in enumerate(pool.imap_unordered(universal_worker, tasks)):
            if score is not None: scores.append(score)
            print_progress_bar(i + 1, len(tasks), prefix=f'Player Analysis ({analysis_params["name"]}):', suffix='Complete', length=50)
    return (np.mean(scores) if scores else 0.0), len(scores)

def estimate_rating_from_quality(model, quality_score):
    """Estimates Elo by finding the rating that maps to the given quality score."""
    search_ratings = np.arange(1350, 3151, 1)
    predicted_qualities = model.predict(search_ratings.reshape(-1, 1))
    return int(search_ratings[np.argmin(np.abs(predicted_qualities - quality_score))])

def save_session(session_data, pgn_path):
    """Saves the current analysis session to a JSON file."""
    session_file = SESSION_FOLDER / pgn_path.with_suffix('.session.json').name
    try:
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=4)
        logging.info(f"Session progress saved to {session_file}")
    except Exception as e:
        logging.error(f"Failed to save session file: {e}")

def load_session(pgn_path):
    """Loads a previous analysis session from a JSON file."""
    session_file = SESSION_FOLDER / pgn_path.with_suffix('.session.json').name
    if session_file.exists():
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
                logging.info(f"Loaded session from {session_file}")
                return data
        except json.JSONDecodeError:
            logging.error(f"Could not parse session file {session_file}. Starting fresh.")
            return None
    return None

def run_full_analysis_for_game(game, ground_truth_template, model_engines, oracle_engine, analysis_params, session_data, pgn_path):
    """A wrapper to run the model building and player analysis for one set of parameters."""
    model_bundle = build_model(ground_truth_template, model_engines, session_data, pgn_path, analysis_params)
    if not model_bundle:
        return None

    avg_score, moves_counted = analyze_player(game, ground_truth_template, oracle_engine, analysis_params)
    if moves_counted == 0:
        return None

    model_bundle['avg_score'] = avg_score
    model_bundle['moves_counted'] = moves_counted
    model_bundle['estimated_rating'] = estimate_rating_from_quality(model_bundle['model'], avg_score)
    model_bundle['analysis_params_name'] = analysis_params['name']

    return model_bundle

# --- MAIN EXECUTION LOGIC ---
def main():
    """Main function to run the analysis process."""
    all_engines = []
    try:
        with open(ENGINES_CSV_PATH, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                row['rating'] = int(row['rating'])
                all_engines.append(row)
    except Exception as e:
        logging.error(f"FATAL: Error reading engines CSV at {ENGINES_CSV_PATH}: {e}"); return

    if not all_engines:
        logging.error(f"FATAL: No engines loaded from {ENGINES_CSV_PATH}. Please check the file."); return

    oracle_engines, model_engines = all_engines[:NUM_ORACLE_ENGINES], all_engines[NUM_ORACLE_ENGINES:]
    logging.info(f"Loaded {len(oracle_engines)} oracle engine(s) and {len(model_engines)} model-building engines.")

    pgn_path_str = input("Enter the full path to your PGN file: ")
    pgn_path = Path(pgn_path_str.strip().strip('"'))
    if not pgn_path.is_file():
        logging.error(f"PGN file not found at {pgn_path}"); return

    session_data = load_session(pgn_path)
    if not session_data:
        session_data = {'pgn_file': str(pgn_path), 'completed_games_data': [], 'in_progress_game': {}}
        # ... (PGN reading logic remains the same) ...
        try:
            with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
                offsets = []
                while True:
                    try:
                        offset = pgn_file.tell()
                        headers = chess.pgn.read_headers(pgn_file)
                        if headers is None: break
                        if PLAYER_TO_ANALYZE.lower() in (headers.get("White", "?").lower(), headers.get("Black", "?").lower()):
                            offsets.append(offset)
                    except Exception: break
                session_data['games_to_process_indices'] = offsets
            save_session(session_data, pgn_path)
        except Exception as e:
            logging.error(f"Could not read PGN file {pgn_path}: {e}"); return

    logging.info(f"Found {len(session_data.get('games_to_process_indices', []))} games remaining to analyze for player {PLAYER_TO_ANALYZE}.")
    report_file_path = SESSION_FOLDER / f"report_{PLAYER_TO_ANALYZE}.txt"
    # ... (Report file setup remains the same) ...

    # Define the two analysis strategies
    analysis_strategies = {
        'depth_first': {'name': 'depth_first', 'depth': PRIMARY_MODEL_BUILD_DEPTH, 'time': PRIMARY_MODEL_BUILD_TIMEOUT},
        'time_first': {'name': 'time_first', 'depth': SECONDARY_MODEL_BUILD_DEPTH, 'time': SECONDARY_MODEL_BUILD_TIMEOUT}
    }

    continuous_mode = False
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        while session_data.get('games_to_process_indices'):
            offset = session_data['games_to_process_indices'][0]
            pgn_file.seek(offset)
            game = chess.pgn.read_game(pgn_file)
            # ... (Game loading and header logic remains the same) ...
            if game is None:
                logging.warning(f"Could not read game at offset {offset}. Skipping.")
                session_data['games_to_process_indices'].pop(0); continue
            game.offset = offset
            if session_data.get('in_progress_game', {}).get('offset') != offset:
                session_data['in_progress_game'] = {'offset': offset}

            white, black = game.headers.get('White', '?'), game.headers.get('Black', '?')
            game_num = len(session_data.get('completed_games_data', [])) + 1
            logging.info(f"--- Starting Analysis for Game {game_num}: {white} vs. {black} ---")

            ground_truth_template = get_ground_truth_template(game, oracle_engines, session_data, pgn_path)
            if not ground_truth_template:
                logging.warning("Failed to generate ground truth template. Skipping game.")
                session_data['games_to_process_indices'].pop(0); save_session(session_data, pgn_path); continue

            # --- ADAPTIVE ANALYSIS LOGIC ---
            # 1. Run primary (depth-first) analysis
            primary_results = run_full_analysis_for_game(game, ground_truth_template, model_engines, oracle_engines[0], analysis_strategies['depth_first'], session_data, pgn_path)
            
            if not primary_results:
                logging.error("Primary analysis failed. Skipping game.")
                session_data['games_to_process_indices'].pop(0); save_session(session_data, pgn_path); continue

            best_results = primary_results
            decision_log = f"Primary analysis (depth-first) chosen with R²={primary_results['r_squared']:.4f}."

            # 2. If R² is low, run secondary (time-first) analysis
            if primary_results['r_squared'] < R_SQUARED_THRESHOLD:
                logging.info(f"Primary R² ({primary_results['r_squared']:.4f}) is below threshold ({R_SQUARED_THRESHOLD}). Running secondary analysis.")
                secondary_results = run_full_analysis_for_game(game, ground_truth_template, model_engines, oracle_engines[0], analysis_strategies['time_first'], session_data, pgn_path)

                if secondary_results and secondary_results['r_squared'] > primary_results['r_squared']:
                    best_results = secondary_results
                    decision_log = f"Secondary analysis (time-first) chosen. R² improved from {primary_results['r_squared']:.4f} to {secondary_results['r_squared']:.4f}."
                elif secondary_results:
                    decision_log = f"Primary analysis (depth-first) kept. Secondary R² ({secondary_results['r_squared']:.4f}) did not improve."
                else:
                    decision_log = "Secondary analysis failed. Keeping primary results."
            
            logging.info(decision_log)
            # --- END ADAPTIVE ANALYSIS LOGIC ---

            # Unpack the best results for reporting
            model, r_squared, ratings, quality_scores, model_type, raw_model_results, avg_score, moves_counted, est_rating, analysis_name = (
                best_results['model'], best_results['r_squared'], best_results['ratings'], best_results['quality_scores'],
                best_results['model_type'], best_results['raw_model_results'], best_results['avg_score'],
                best_results['moves_counted'], best_results['estimated_rating'], best_results['analysis_params_name']
            )

            # Plotting and reporting using the 'best_results'
            plt.figure(figsize=(10, 6))
            plt.scatter(ratings.flatten(), quality_scores, alpha=0.7, label="Engine Performance")
            plot_x = np.linspace(ratings.min(), ratings.max(), 200).reshape(-1, 1)
            plt.plot(plot_x, model.predict(plot_x), color='red', lw=2, label=f"{model_type} Model (R²={r_squared:.4f})")
            plt.axhline(y=avg_score, color='g', linestyle='--', label=f"Player Score ({avg_score:.4f}) -> {est_rating} Elo")
            plt.title(f"Game {game_num}: {white} vs. {black} (Analysis: {analysis_name})"); plt.xlabel("Engine Elo Rating (from CSV)"); plt.ylabel("Weighted Score"); plt.grid(True); plt.legend()
            graph_path = SESSION_FOLDER / f"performance_graph_game_{game_num}.png"
            plt.savefig(graph_path); plt.close()

            game_summary = f"Game {game_num}: {white} vs. {black} -> Est. Rating: {est_rating} (Model: {model_type}, R²={r_squared:.3f}, Moves: {moves_counted})"
            logging.info(game_summary)
            with open(report_file_path, 'a', encoding='utf-8') as report_file:
                report_file.write(game_summary + "\n")
                report_file.write(f"  > Decision: {decision_log}\n")

            # Save results to session file
            session_data.setdefault('completed_games_data', []).append({
                'game_num': game_num, 'white': white, 'black': black, 'r_squared': r_squared,
                'graph_path': str(graph_path), 'model_type': model_type, 'avg_score': avg_score,
                'moves': moves_counted, 'estimated_rating': est_rating,
                'model_results_raw': raw_model_results, 'analysis_decision': decision_log
            })
            session_data['games_to_process_indices'].pop(0)
            session_data['in_progress_game'] = {}
            save_session(session_data, pgn_path)
            logging.info(f"--- Finished Analysis for Game {game_num}. Estimated rating: {est_rating}. ---")

            if not continuous_mode and session_data['games_to_process_indices']:
                try:
                    user_input = input("Continue with next game? (y/n/c for yes/no/continuous): ").lower().strip()
                    if user_input == 'n': break
                    if user_input == 'c': continuous_mode = True
                except EOFError:
                    logging.info("EOFError detected. Exiting.")
                    break

    logging.info("All games analyzed.")
    completed_games = session_data.get('completed_games_data', [])
    if completed_games:
        final_avg_rating = int(np.mean([g['estimated_rating'] for g in completed_games]))
        summary = f"\nFinal Average Estimated Rating: {final_avg_rating} Elo across {len(completed_games)} games."
        logging.info(summary)
        with open(report_file_path, 'a', encoding='utf-8') as report_file:
            report_file.write("-" * 50 + "\n")
            report_file.write(summary + "\n")


if __name__ == "__main__":
    if sys.platform.startswith('darwin') or sys.platform.startswith('win'):
        multiprocessing.set_start_method('spawn', force=True)
    main()
