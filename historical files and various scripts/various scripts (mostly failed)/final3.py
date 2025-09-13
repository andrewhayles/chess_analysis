# chess_framework_analyzer.py
# This script analyzes PGN chess games to test different move quality scoring frameworks.
# It determines which framework's scores best correlate with engine Elo ratings by comparing R² values.
#
# KEY CHANGES:
#   - Fixed the analysis loop to use a time-only limit, ensuring engines analyze for the
#     configured duration instead of exiting instantly on a shallow depth.
#   - Adjusted the default model-building timeout to a more practical value.
#
# Frameworks Implemented:
#   1. RWPL (Refined Win-Probability-Loss): Measures performance based on the loss of win probability.
#   2. CAI (Context-Aware Impact): Weights win probability loss by the criticality of the position.
#   3. Complexity: Measures the standard deviation of evaluations of the top N moves.

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
import subprocess

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(__file__).resolve().parent
# IMPORTANT: You must create a 'real_engines.csv' file in the same directory.
# It needs columns: name, path, rating, uci_options (as JSON string)
ENGINES_CSV_PATH = PROJECT_FOLDER / "real_engines.csv"
SESSION_FOLDER = PROJECT_FOLDER / "chess_analysis_session"
SESSION_FOLDER.mkdir(exist_ok=True)

PLAYER_TO_ANALYZE = "Desjardins373" # <--- CHANGE THIS
NUM_ANALYSIS_CORES = 2 # Adjust based on your CPU
NUM_ORACLE_ENGINES = 1 # The strongest engine(s) to establish "ground truth"
MODEL_BUILD_ENGINES = 32 # Number of engines to use for building the regression model

# --- ANALYSIS SETTINGS ---
# Oracle engine settings (for establishing ground truth)
ORACLE_ANALYSIS_DEPTH = 22
ORACLE_TIMEOUT = 6000

# Model-building engine settings
MODEL_BUILD_TIMEOUT = 300 # CHANGED: From 120 to 5 seconds for a more practical analysis time per move.

# --- FRAMEWORK PARAMETERS ---
# These are the tuning parameters from your document.
K1_WPL = 0.00368  # For Win Probability calculation
K2_RWPL = 0.5     # For RWPL score curve (0.5 = sqrt, 1 = linear, 2 = quadratic)
K3_CAI = 0.004     # For Criticality calculation
W_CAI = 1.0       # Weight for criticality in CAI score

# --- SCRIPT SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(SESSION_FOLDER / 'analysis_log.txt'), logging.StreamHandler(sys.stdout)]
)

# --- SCORING FRAMEWORK IMPLEMENTATIONS ---

def convert_eval_to_win_prob(evaluation_cp):
    """Converts a centipawn evaluation to a Win Probability using the specified sigmoid function."""
    if evaluation_cp is None:
        return 0.5
    return 1 / (1 + np.exp(-K1_WPL * evaluation_cp))

def calculate_rwpl_score(wp_loss):
    """Calculates the Refined Win-Probability-Loss (RWPL) score for a single move."""
    capped_wp_loss = min(wp_loss, 0.5)
    return 100 * (1 - 2 * (capped_wp_loss ** K2_RWPL))

def calculate_cai_score(wp_loss, criticality):
    """Calculates the Context-Aware Impact (CAI) score for a single move."""
    impact = wp_loss * (1 + W_CAI * criticality)
    capped_impact = min(impact, 0.5)
    return 100 * (1 - 2 * (capped_impact ** K2_RWPL))

def calculate_move_scores(best_move_eval, played_move_eval, second_best_move_eval, top_n_evals):
    """Calculates the scores for a single move across all implemented frameworks."""
    if played_move_eval is None or best_move_eval is None:
        return {}

    wp_best = convert_eval_to_win_prob(best_move_eval)
    wp_played = convert_eval_to_win_prob(played_move_eval)
    wp_loss = max(0, wp_best - wp_played)

    scores = {'RWPL': calculate_rwpl_score(wp_loss)}

    criticality = 0.0
    if second_best_move_eval is not None:
        eval_gap = abs(best_move_eval - second_best_move_eval)
        criticality = 1 / (1 + np.exp(-K3_CAI * eval_gap))

    scores['CAI'] = calculate_cai_score(wp_loss, criticality)
    scores['Impact'] = scores['CAI']

    scores['Complexity'] = np.std(top_n_evals) if top_n_evals and len(top_n_evals) > 1 else 0

    return scores

# --- CORE ANALYSIS SCRIPT (MODIFIED FOR FRAMEWORK EVALUATION) ---

def get_standard_engine_options(csv_options_str, engine_info):
    """Parses engine options from CSV and adds standard configurations conditionally."""
    options = {}
    if 'stockfish' in engine_info['name'].lower():
        options["Hash"] = 256
    if csv_options_str:
        try:
            options.update(json.loads(csv_options_str))
        except json.JSONDecodeError:
            logging.warning(f"Could not decode JSON options from CSV: {csv_options_str}")
    return options

def print_progress_bar(iteration, total, prefix='', suffix='', length=50):
    """Displays a terminal progress bar."""
    if total == 0: total = 1
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: sys.stdout.write('\n')

def save_session(session_data, pgn_path):
    session_file = SESSION_FOLDER / pgn_path.with_suffix('.session.json').name
    with open(session_file, 'w') as f: json.dump(session_data, f, indent=2)

def load_session(pgn_path):
    session_file = SESSION_FOLDER / pgn_path.with_suffix('.session.json').name
    if session_file.exists():
        try:
            with open(session_file, 'r') as f:
                logging.info(f"Loaded session from {session_file}")
                return json.load(f)
        except json.JSONDecodeError:
            logging.error("Could not parse session file. Starting fresh.")
    return None

def get_sparse_engines(full_engine_list, num_points, exclude_engines=None):
    """Selects a subset of engines with evenly spaced ratings."""
    if exclude_engines is None: exclude_engines = []
    exclude_names = {e['name'] for e in exclude_engines}
    available_engines = [e for e in full_engine_list if e['name'] not in exclude_names]
    if len(available_engines) <= num_points: return available_engines
    sorted_available = sorted(available_engines, key=lambda x: x['rating'])
    indices = np.linspace(0, len(sorted_available) - 1, num_points, dtype=int)
    return [sorted_available[i] for i in indices]

def universal_worker(args):
    """A universal worker for all analysis types."""
    (fen, engine_info, limit_params, worker_mode) = args
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path'], stderr=subprocess.DEVNULL) as engine:
            options = get_standard_engine_options(engine_info.get('uci_options', '{}'), engine_info)
            engine.configure(options)
            board = chess.Board(fen)

            if worker_mode == 'oracle':
                info = engine.analyse(board, chess.engine.Limit(**limit_params), multipv=5)
                template = []
                if not info: return None
                pov_color = board.turn
                for move_info in info:
                    score_obj = move_info.get('score')
                    if score_obj and 'pv' in move_info:
                        pov_score = score_obj.pov(pov_color).score(mate_score=30000)
                        if pov_score is not None:
                            template.append({'move': move_info['pv'][0].uci(), 'eval': pov_score})
                return (fen, template)

            elif worker_mode == 'model_simulation':
                result = engine.play(board, chess.engine.Limit(**limit_params))
                if not result.move: return None
                analysis = engine.analyse(board, chess.engine.Limit(**limit_params), root_moves=[result.move])
                if not analysis: return None
                pov_score = analysis['score'].pov(board.turn).score(mate_score=30000)
                if pov_score is None: return None
                return (fen, engine_info['name'], {'move': result.move.uci(), 'eval': pov_score})
    except Exception as e:
        logging.error(f"Worker '{worker_mode}' for {engine_info['name']} on FEN '{fen}' failed: {e}")
    return None

def get_ground_truth_template(game, oracle_engine, session_data, pgn_path):
    """Generates the 'answer key' of best moves and their evaluations."""
    if 'ground_truth_template' in session_data.get('in_progress_game', {}):
        logging.info("Loading ground truth template from session.")
        return session_data['in_progress_game']['ground_truth_template']

    logging.info(f"Generating ground truth template with {oracle_engine['name']}...")
    player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    fens_to_analyze = [
        node.board().fen() for i, node in enumerate(game.mainline())
        if node.board().turn == player_color and i > 20 and i % 5 == 0
    ]
    if not fens_to_analyze: return {}

    template = {}
    limit = {'depth': ORACLE_ANALYSIS_DEPTH, 'time': ORACLE_TIMEOUT}
    tasks = [(fen, oracle_engine, limit, 'oracle') for fen in fens_to_analyze]
    
    print_progress_bar(0, len(tasks), prefix='Truth Template:')
    with multiprocessing.Pool(processes=NUM_ORACLE_ENGINES) as pool:
        for i, result in enumerate(pool.imap_unordered(universal_worker, tasks)):
            if result:
                fen, move_template = result
                template[fen] = move_template
            print_progress_bar(i + 1, len(tasks), prefix='Truth Template:')

    session_data['in_progress_game']['ground_truth_template'] = template
    save_session(session_data, pgn_path)
    return template

def calculate_framework_models(simulation_results, ground_truth_template, model_engines):
    """Calculates regression models and R² for each framework based on current data."""
    engine_scores_by_framework = defaultdict(lambda: defaultdict(list))
    for fen, template in ground_truth_template.items():
        if not template: continue
        best_move_eval = template[0]['eval']
        second_best_move_eval = template[1]['eval'] if len(template) > 1 else None
        top_n_evals = [m['eval'] for m in template]

        if fen not in simulation_results: continue
        for engine_name, move_data in simulation_results[fen].items():
            played_move_eval = move_data['eval']
            move_scores = calculate_move_scores(best_move_eval, played_move_eval, second_best_move_eval, top_n_evals)
            for framework_name, score in move_scores.items():
                engine_scores_by_framework[framework_name][engine_name].append(score)

    avg_scores_by_framework = defaultdict(dict)
    for framework, engine_scores in engine_scores_by_framework.items():
        for engine, scores in engine_scores.items():
            avg_scores_by_framework[framework][engine] = np.mean(scores) if scores else 0

    framework_models = {}
    for framework, scores_dict in avg_scores_by_framework.items():
        ratings_data, scores_data = [], []
        for eng_info in model_engines:
            if eng_info['name'] in scores_dict:
                ratings_data.append(eng_info['rating'])
                scores_data.append(scores_dict[eng_info['name']])
        
        if len(ratings_data) < 3: continue
        
        ratings = np.array(ratings_data).reshape(-1, 1)
        scores = np.array(scores_data)
        model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(ratings, scores)
        r_squared = r2_score(scores, model.predict(ratings))
        
        framework_models[framework] = {'model': model, 'r_squared': r_squared, 'ratings': ratings, 'scores': scores}
        
    return framework_models

def log_live_update(simulation_results, ground_truth_template, all_model_engines, completed_engine_name):
    """Logs a live leaderboard of framework performance."""
    processed_engine_names = set()
    for fen_results in simulation_results.values():
        processed_engine_names.update(fen_results.keys())

    completed_engines = [e for e in all_model_engines if e['name'] in processed_engine_names]

    if len(completed_engines) < 3:
        logging.info(f"Completed {completed_engine_name}. Need at least 3 engine results to calculate R². ({len(completed_engines)}/{len(all_model_engines)})")
        return

    framework_models = calculate_framework_models(simulation_results, ground_truth_template, completed_engines)
    if not framework_models: return

    sorted_frameworks = sorted(framework_models.items(), key=lambda item: item[1]['r_squared'], reverse=True)

    logging.info(f"\n--- LIVE UPDATE after '{completed_engine_name}' ({len(completed_engines)}/{len(all_model_engines)} engines) ---")
    logging.info("Current Framework R² Leaderboard:")
    for name, data in sorted_frameworks:
        logging.info(f"  - {name:<12}: R² = {data['r_squared']:.4f}")
    logging.info("-" * 50)

def run_engine_simulations(ground_truth_template, engines, session_data, pgn_path):
    """Runs simulations and provides live updates after each engine."""
    sim_results = session_data.get('in_progress_game', {}).get('simulation_results', defaultdict(dict))

    for engine_idx, engine_info in enumerate(engines):
        engine_name = engine_info['name']
        is_complete = all(engine_name in sim_results.get(fen, {}) for fen in ground_truth_template.keys())
        if is_complete:
            logging.info(f"({engine_idx+1}/{len(engines)}) Skipping {engine_name}, results already in session.")
            continue

        logging.info(f"--- ({engine_idx+1}/{len(engines)}) Simulating with {engine_name} ---")
        # FIX: Use a time-only limit. The previous depth limit was met instantly.
        limit = {'time': MODEL_BUILD_TIMEOUT}
        tasks = [(fen, engine_info, limit, 'model_simulation') for fen in ground_truth_template]
        
        print_progress_bar(0, len(tasks), prefix=f'Simulating ({engine_name}):')
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            for i, result in enumerate(pool.imap_unordered(universal_worker, tasks)):
                if result:
                    fen, _, move_data = result
                    if fen not in sim_results: sim_results[fen] = {}
                    sim_results[fen][engine_name] = move_data
                print_progress_bar(i + 1, len(tasks), prefix=f'Simulating ({engine_name}):')
        
        session_data['in_progress_game']['simulation_results'] = dict(sim_results)
        save_session(session_data, pgn_path)
        log_live_update(sim_results, ground_truth_template, engines, engine_info['name'])
    
    return sim_results

def report_final_evaluation(framework_models):
    """Logs the final summary of the framework comparison."""
    if not framework_models:
        logging.error("Could not build any models for final evaluation.")
        return None, None

    logging.info("\n--- FINAL Framework R² Comparison ---")
    sorted_frameworks = sorted(framework_models.items(), key=lambda item: item[1]['r_squared'], reverse=True)
    for name, data in sorted_frameworks:
        logging.info(f"  - {name:<12}: R² = {data['r_squared']:.4f}")
    
    best_framework_name = sorted_frameworks[0][0]
    logging.info(f"--- Best Framework: {best_framework_name} (R² = {framework_models[best_framework_name]['r_squared']:.4f}) ---\n")
    
    return best_framework_name, framework_models[best_framework_name]

def main():
    """Main function to run the analysis process."""
    all_engines = []
    try:
        with open(ENGINES_CSV_PATH, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if row['name'].strip().startswith('#'): continue
                row['rating'] = int(row['rating'])
                all_engines.append(row)
    except Exception as e:
        logging.error(f"FATAL: Error reading {ENGINES_CSV_PATH}: {e}"); return

    if not all_engines:
        logging.error(f"FATAL: No engines loaded. Check {ENGINES_CSV_PATH}."); return

    all_engines.sort(key=lambda x: x['rating'], reverse=True)
    oracle_engine = all_engines[0]
    model_engines_full = all_engines[NUM_ORACLE_ENGINES:]
    
    model_engines = get_sparse_engines(model_engines_full, MODEL_BUILD_ENGINES, exclude_engines=[oracle_engine])
    logging.info(f"Oracle: {oracle_engine['name']}. Model builders: {len(model_engines)} engines.")

    pgn_path_str = input("Enter the full path to your PGN file: ")
    pgn_path = Path(pgn_path_str.strip().strip('"'))
    if not pgn_path.is_file():
        logging.error(f"PGN file not found at {pgn_path}"); return

    session_data = load_session(pgn_path) or {'pgn_file': str(pgn_path)}
    session_data['in_progress_game'] = session_data.get('in_progress_game', {})

    try:
        with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
            game = chess.pgn.read_game(pgn_file)
            while game:
                if PLAYER_TO_ANALYZE.lower() in (game.headers.get("White", "?").lower(), game.headers.get("Black", "?").lower()):
                    break
                game = chess.pgn.read_game(pgn_file)
            if not game:
                logging.error(f"No games found for player {PLAYER_TO_ANALYZE} in PGN."); return
    except Exception as e:
        logging.error(f"Could not read PGN file {pgn_path}: {e}"); return

    white, black = game.headers.get('White', '?'), game.headers.get('Black', '?')
    logging.info(f"--- Starting Analysis for Game: {white} vs. {black} ---")

    ground_truth = get_ground_truth_template(game, oracle_engine, session_data, pgn_path)
    if not ground_truth:
        logging.error("Failed to generate ground truth template. Exiting."); return

    sim_results = run_engine_simulations(ground_truth, model_engines, session_data, pgn_path)
    if not sim_results:
        logging.error("Failed to run engine simulations. Exiting."); return

    final_models = calculate_framework_models(sim_results, ground_truth, model_engines)
    best_framework, final_model_bundle = report_final_evaluation(final_models)
    if not best_framework:
        logging.error("Framework evaluation failed. Exiting."); return

    plt.figure(figsize=(12, 7))
    plt.scatter(final_model_bundle['ratings'].flatten(), final_model_bundle['scores'], alpha=0.7, label="Engine Performance")
    plot_x = np.linspace(final_model_bundle['ratings'].min(), final_model_bundle['ratings'].max(), 200).reshape(-1, 1)
    plt.plot(plot_x, final_model_bundle['model'].predict(plot_x), color='red', lw=2, label=f"Polyfit Model (R²={final_model_bundle['r_squared']:.4f})")
    plt.title(f"Best Framework: '{best_framework}' Performance vs. Engine Rating")
    plt.xlabel("Engine Elo Rating (from CSV)")
    plt.ylabel(f"Average '{best_framework}' Score")
    plt.grid(True)
    plt.legend()
    graph_path = SESSION_FOLDER / f"framework_comparison_graph.png"
    plt.savefig(graph_path)
    plt.close()

    logging.info(f"Analysis complete. Graph for best framework saved to {graph_path}")

if __name__ == "__main__":
    if sys.platform.startswith('darwin') or sys.platform.startswith('win'):
        multiprocessing.set_start_method('spawn', force=True)
    main()
