# optimizer_v4.py
# A specialized script to find the optimal analysis settings for the main rating script.
#
# HOW IT WORKS:
# 1. It analyzes a specified number of games (e.g., the first 3) from the PGN file, one by one.
# 2. For each game, it generates a ground truth template and runs the full optimization process.
# 3. It logs the best settings for EACH individual game.
# 4. It saves all raw results (every timeout/template combo for every game) to a detailed CSV file.
# 5. After all games are processed, it calculates the AVERAGE R-squared across all games for each setting.
# 6. Finally, it reports the combination of timeout and template size that has the best average performance.

import chess
import chess.engine
import chess.pgn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from pathlib import Path
import logging
import multiprocessing
import json
import sys
import csv
from collections import defaultdict

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(__file__).resolve().parent
ENGINES_CSV_PATH = PROJECT_FOLDER / "real_engines.csv"
SESSION_FOLDER = PROJECT_FOLDER / "chess_analysis_session_optimizer"
SESSION_FOLDER.mkdir(exist_ok=True)

# --- OPTIMIZATION SETTINGS ---
# Set the number of games to use for the optimization process.
NUM_GAMES_TO_OPTIMIZE = 3
# Define the different timeout values (in seconds) to test.
TIMEOUTS_TO_TEST = [.1, .5, 2, 6, 15, 30, 45, 60, 120, 180]
# Define the template sizes to test for each timeout.
TEMPLATE_SIZES_TO_TEST = [1, 2, 3, 4, 5]
# Define a fixed depth for this optimization test, or None to use time-only.
OPTIMIZATION_DEPTH = 12

# --- FIXED ORACLE SETTINGS (for generating the ground truth) ---
ORACLE_ANALYSIS_DEPTH = 20
ORACLE_TIMEOUT = 300 # A high value to ensure quality
ORACLE_TEMPLATE_MOVE_COUNT = 5 # Must be >= max(TEMPLATE_SIZES_TO_TEST)

# --- GENERAL SETTINGS ---
PLAYER_TO_ANALYZE = "Desjardins373" # Player name to find in PGN
NUM_ANALYSIS_CORES = 2
NUM_ORACLE_ENGINES = 1
HASH_SIZE_MB = 256
TABLEBASE_PATH = r"" # Optional path to Syzygy tablebases

# --- SCRIPT SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(SESSION_FOLDER / 'optimizer_log.txt'), logging.StreamHandler(sys.stdout)]
)

if not ENGINES_CSV_PATH.is_file():
    logging.error(f"FATAL: Engines CSV file not found at '{ENGINES_CSV_PATH}'")
    sys.exit()

# --- HELPER & WORKER FUNCTIONS (Aligned with finalwork script) ---

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

def universal_worker(args):
    """A universal worker for oracle and model simulation modes."""
    (fen, engine_info, limit_params, worker_mode) = args

    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            options = get_standard_engine_options(engine_info.get('uci_options', '{}'))
            engine.configure(options)
            board = chess.Board(fen)

            if worker_mode == 'oracle':
                info = engine.analyse(board, chess.engine.Limit(**limit_params), multipv=ORACLE_TEMPLATE_MOVE_COUNT)
                template = []
                if not info: return None
                pov_color = board.turn
                best_pov_score = info[0]['score'].pov(pov_color).score(mate_score=30000)
                if best_pov_score is None: return None
                for move_info in info:
                    if 'pv' in move_info and 'score' in move_info:
                        move = move_info['pv'][0]
                        current_pov_score = move_info['score'].pov(pov_color).score(mate_score=30000)
                        if current_pov_score is None: continue
                        cpl = max(0, best_pov_score - current_pov_score)
                        quality_score = calculate_move_quality(cpl)
                        template.append({'move': move.uci(), 'score': quality_score})
                return template

            elif worker_mode == 'model_simulation':
                result = engine.play(board, chess.engine.Limit(**limit_params))
                return result.move.uci() if result.move else None

    except Exception as e:
        logging.error(f"Worker in mode '{worker_mode}' for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
    return None

def get_ground_truth_template(games, oracle_engines):
    """Generates the 'answer key' of best moves for a list of games."""
    all_fens_to_analyze = []
    for game in games:
        player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
        fens_for_game = [node.board().fen() for node in game.mainline() if node.board().turn == player_color]
        all_fens_to_analyze.extend(fens_for_game)
    
    fens_to_analyze = sorted(list(set(all_fens_to_analyze)))
    logging.info(f"Found {len(fens_to_analyze)} unique positions to analyze across {len(games)} game(s).")
    if not fens_to_analyze: return {}

    template = {}
    limit = {'depth': ORACLE_ANALYSIS_DEPTH, 'time': ORACLE_TIMEOUT}
    tasks = [(fen, oracle_engines[0], limit, 'oracle') for fen in fens_to_analyze]
    print_progress_bar(0, len(tasks), prefix=f'Truth Template ({oracle_engines[0]["name"]}):', suffix='Complete', length=50)
    with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
        for i, result in enumerate(pool.imap_unordered(universal_worker, tasks)):
            if result: template[tasks[i][0]] = result
            print_progress_bar(i + 1, len(tasks), prefix=f'Truth Template ({oracle_engines[0]["name"]}):', suffix='Complete', length=50)
    return template

def run_engine_simulations_for_timeout(ground_truth_template, model_engines, timeout):
    """Runs engine simulations for a specific timeout and returns the collected moves."""
    logging.info(f"----- TESTING TIMEOUT: {timeout} seconds -----")
    limit = {'depth': OPTIMIZATION_DEPTH, 'time': timeout}
    
    all_tasks = []
    for fen in ground_truth_template.keys():
        for engine_info in model_engines:
            all_tasks.append((fen, engine_info, limit, 'model_simulation'))

    prefix = f'Sims @ {timeout}s:'
    print_progress_bar(0, len(all_tasks), prefix=prefix, suffix='Complete', length=50)
    
    simulation_results = defaultdict(dict)
    with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
        for i, move_uci in enumerate(pool.imap_unordered(universal_worker, all_tasks)):
            task_fen, task_engine_name = all_tasks[i][0], all_tasks[i][1]['name']
            if move_uci: simulation_results[task_fen][task_engine_name] = move_uci
            print_progress_bar(i + 1, len(all_tasks), prefix=prefix, suffix='Complete', length=50)
    return dict(simulation_results)

def calculate_scores_from_sims(simulation_results, ground_truth_template, template_size):
    """Calculates average scores for each engine based on a given template size."""
    engine_scores = defaultdict(list)
    for fen, moves_by_engine in simulation_results.items():
        if fen not in ground_truth_template: continue
        template_moves = {entry['move']: entry['score'] for entry in ground_truth_template[fen][:template_size]}
        for engine_name, move_uci in moves_by_engine.items():
            engine_scores[engine_name].append(template_moves.get(move_uci, 0.0))
    return {name: np.mean(scores) if scores else 0.0 for name, scores in engine_scores.items()}

def build_model_from_scores(engine_scores, engines_to_process):
    """Builds a regression model and returns its R-squared value."""
    ratings_data, quality_scores_data = [], []
    for eng_info in engines_to_process:
        if eng_info['name'] in engine_scores:
            ratings_data.append(eng_info['rating'])
            quality_scores_data.append(engine_scores[eng_info['name']])

    if len(ratings_data) < 3: return -1.0
    
    ratings = np.array(ratings_data).reshape(-1, 1)
    quality_scores = np.array(quality_scores_data)
    
    linear_model = LinearRegression().fit(ratings, quality_scores)
    poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(ratings, quality_scores)
    
    test_ratings = np.arange(ratings.min(), ratings.max(), 10).reshape(-1, 1)
    poly_predictions = poly_model.predict(test_ratings)
    is_monotonic = np.all(np.diff(poly_predictions) >= 0)

    linear_r2 = r2_score(quality_scores, linear_model.predict(ratings))
    poly_r2 = r2_score(quality_scores, poly_model.predict(ratings)) if is_monotonic else -1.0
    return max(linear_r2, poly_r2)

def run_optimization_for_game(game, model_engines, oracle_engines):
    """Runs the full optimization process for a single game and returns the results."""
    ground_truth_template = get_ground_truth_template([game], oracle_engines)
    if not ground_truth_template:
        logging.error("Failed to generate ground truth for the game. Skipping.")
        return []

    game_results = []
    for timeout in TIMEOUTS_TO_TEST:
        simulation_results = run_engine_simulations_for_timeout(ground_truth_template, model_engines, timeout)
        for t_size in TEMPLATE_SIZES_TO_TEST:
            scores = calculate_scores_from_sims(simulation_results, ground_truth_template, t_size)
            r_squared = build_model_from_scores(scores, model_engines)
            logging.info(f"  - Result for Timeout: {timeout}s, Template Size: {t_size} -> R-squared: {r_squared:.4f}")
            game_results.append({'timeout': timeout, 'template_size': t_size, 'r2': r_squared})
    return game_results

def process_and_log_combined_results(all_results_by_game):
    """Aggregates results from all games, logs them, and saves a detailed CSV."""
    logging.info("\n\n----- COMBINED OPTIMIZATION RESULTS -----")

    # Write detailed CSV file with raw results from every game
    csv_path = SESSION_FOLDER / "optimization_details.csv"
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['game_id', 'timeout', 'template_size', 'r_squared'])
            for game_id, results in all_results_by_game.items():
                for res in results:
                    writer.writerow([game_id, res['timeout'], res['template_size'], f"{res['r2']:.6f}"])
        logging.info(f"Full detailed results saved to {csv_path}")
    except IOError as e:
        logging.error(f"Could not write to CSV file at {csv_path}: {e}")

    # Aggregate results to find the average R² for each setting
    aggregated_scores = defaultdict(list)
    for game_id, results in all_results_by_game.items():
        for res in results:
            key = (res['timeout'], res['template_size'])
            if res['r2'] > -1: # Only include valid scores in average
                aggregated_scores[key].append(res['r2'])

    average_results = []
    for (timeout, t_size), r2_values in aggregated_scores.items():
        if r2_values:
            avg_r2 = np.mean(r2_values)
            average_results.append({'timeout': timeout, 'template_size': t_size, 'avg_r2': avg_r2})

    logging.info("\nAverage R-squared results across all games:")
    for res in sorted(average_results, key=lambda x: (x['timeout'], x['template_size'])):
         logging.info(f"  - Timeout: {res['timeout']:<3}s | Template Size: {res['template_size']} | Avg R² = {res['avg_r2']:.4f}")

    if average_results:
        best_overall_result = max(average_results, key=lambda x: x['avg_r2'])
        best_timeout = best_overall_result['timeout']
        best_t_size = best_overall_result['template_size']
        best_avg_r2 = best_overall_result['avg_r2']
        logging.info(f"\nOptimal Combination (Highest Average): Timeout={best_timeout}s, Template Size={best_t_size} (with Avg R² = {best_avg_r2:.4f})")
        logging.info("RECOMMENDATION: Update the following in your main script:")
        logging.info(f"  - PRIMARY_MODEL_BUILD_TIMEOUT = {best_timeout}")
        logging.info(f"  - PRIMARY_PLAYER_ANALYSIS_TIMEOUT = {best_timeout}")
        logging.info(f"  - TEMPLATE_MOVE_COUNT = {best_t_size}")

# --- MAIN EXECUTION LOGIC ---
def main():
    """Main function to run the optimization process."""
    all_engines = []
    try:
        with open(ENGINES_CSV_PATH, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                row['rating'] = int(row['rating'])
                all_engines.append(row)
    except Exception as e:
        logging.error(f"FATAL: Error reading engines CSV: {e}"); return

    oracle_engines, model_engines = all_engines[:NUM_ORACLE_ENGINES], all_engines[NUM_ORACLE_ENGINES:]
    logging.info(f"Loaded {len(oracle_engines)} oracle engine(s) and {len(model_engines)} model-building engines.")

    pgn_path_str = input("Enter the full path to your PGN file: ")
    pgn_path = Path(pgn_path_str.strip().strip('"'))
    if not pgn_path.is_file():
        logging.error(f"PGN file not found at {pgn_path}"); return

    games_to_optimize = []
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        while len(games_to_optimize) < NUM_GAMES_TO_OPTIMIZE:
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None: 
                    logging.info("Reached end of PGN file.")
                    break
                if PLAYER_TO_ANALYZE.lower() in (game.headers.get("White", "?").lower(), game.headers.get("Black", "?").lower()):
                    games_to_optimize.append(game)
                    logging.info(f"Added game {len(games_to_optimize)}/{NUM_GAMES_TO_OPTIMIZE}: {game.headers.get('White')} vs {game.headers.get('Black')}")
            except Exception as e:
                logging.error(f"Error reading game from PGN: {e}"); break
    
    if not games_to_optimize:
        logging.error(f"No games found for player {PLAYER_TO_ANALYZE} in PGN file. Aborting."); return

    all_results_by_game = {}
    for i, game in enumerate(games_to_optimize):
        white = game.headers.get('White', 'W').replace(' ', '_')
        black = game.headers.get('Black', 'B').replace(' ', '_')
        game_id = f"Game{i+1}_{white}_vs_{black}"
        logging.info(f"\n--- Starting Optimization for {game_id} ---")

        game_results = run_optimization_for_game(game, model_engines, oracle_engines)
        if game_results:
            all_results_by_game[game_id] = game_results
            best_for_game = max(game_results, key=lambda x: x['r2'])
            logging.info(f"--- Best for {game_id}: Timeout={best_for_game['timeout']}s, Template={best_for_game['template_size']}, R²={best_for_game['r2']:.4f} ---")

    if all_results_by_game:
        process_and_log_combined_results(all_results_by_game)
    else:
        logging.warning("No optimization results were generated for any game.")

if __name__ == "__main__":
    if sys.platform.startswith('darwin') or sys.platform.startswith('win'):
        multiprocessing.set_start_method('spawn', force=True)
    main()
