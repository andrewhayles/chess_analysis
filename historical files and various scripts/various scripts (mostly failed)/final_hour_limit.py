# chess_framework_optimizer_v8.py
# This script analyzes PGN chess games to test and optimize move quality scoring frameworks.
#
# --- KEY FEATURES ---
#   - Time-Limited Execution: Runs for a specified duration (e.g., 8.5 hours), then automatically finalizes.
#   - Resumable Sessions: Automatically saves progress after each game to a session file.
#   - Consolidated Logging: All operational logs are stored in 'optimizer_main.log'.
#   - Final Summary Report: After finishing, it generates a 'final_run_summary.txt' with optimized
#     constants, R² leaderboards, and run statistics.
#   - Two Operating Modes:
#     1. ANALYSIS MODE: Analyzes games using the best-known constants.
#     2. OPTIMIZATION MODE: Finds optimal constants for CAI/RWPL and HitCount frameworks.
#
# --- FRAMEWORKS ---
#   - RWPL (Refined Win-Probability-Loss)
#   - CAI (Context-Aware Impact)
#   - Complexity (Corrected logic: now engine-specific)
#   - HitCount_T<N>: A simple template-matching score.

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
import os
import time

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(__file__).resolve().parent
ENGINES_CSV_PATH = PROJECT_FOLDER / "real_engines.csv"
SESSION_FOLDER = PROJECT_FOLDER / "chess_analysis_session"
SESSION_FILE = SESSION_FOLDER / "optimizer_session.json"
CONSTANTS_FILE = SESSION_FOLDER / "optimized_constants.json"
FINAL_SUMMARY_FILE = SESSION_FOLDER / "final_run_summary.txt"
MAIN_LOG_FILE = SESSION_FOLDER / "optimizer_main.log"
SESSION_FOLDER.mkdir(exist_ok=True)

# --- RUN SETTINGS ---
RUN_DURATION_HOURS = 8.5 # Set how long the script should run before finalizing.
PLAYER_TO_ANALYZE = "Desjardins373" # <--- CHANGE THIS
NUM_ANALYSIS_CORES = 2 # Adjust based on your CPU
NUM_ORACLE_ENGINES = 1
MODEL_BUILD_ENGINES = 18

# --- ANALYSIS SETTINGS ---
ORACLE_ANALYSIS_DEPTH = 20
ORACLE_TIMEOUT = 300
MODEL_BUILD_DEPTH = 12
MODEL_BUILD_TIMEOUT = 60
POSITIONS_PER_GAME_TO_SAMPLE = 7

# --- HITCOUNT FRAMEWORK SETTINGS ---
HITCOUNT_MAX_TEMPLATE_SIZE = 5  # Creates HitCount_T1, T2, ... up to this number

# --- DEFAULT FRAMEWORK PARAMETERS ---
DEFAULT_CONSTANTS = {
    "K1_WPL": 0.00368, "K2_RWPL": 0.5, "K3_CAI": 0.004, "W_CAI": 1.0,
    "HITCOUNT_SECOND_MOVE_WEIGHT": 0.75
}

# --- OPTIMIZATION PARAMETERS ---
OPTIMIZATION_CYCLES = 3
OPTIMIZATION_SEARCH_SPACE = {
    "K1_WPL": np.linspace(0.001, 0.008, 15),
    "K2_RWPL": np.linspace(0.3, 1.5, 15),
    "K3_CAI": np.linspace(0.001, 0.01, 15),
    "W_CAI": np.linspace(0.5, 2.5, 15),
    "HITCOUNT_SECOND_MOVE_WEIGHT": np.linspace(0.25, 0.95, 15)
}

# --- SCRIPT SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(MAIN_LOG_FILE), logging.StreamHandler(sys.stdout)]
)

# --- UTILITY & SESSION MANAGEMENT ---

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_json_file(file_path, default_data=None):
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not read or parse {file_path}: {e}. Using default data.")
            return default_data or {}
    return default_data or {}

def save_json_file(data, file_path):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logging.error(f"Failed to save to {file_path}: {e}")

def log_progress_update(leaderboard_text, game_info, session_data):
    """Logs a detailed progress update to the main logger."""
    update_message = (
        f"\n--- PROGRESS UPDATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        f"Last Game Processed: {game_info['id']} ({game_info['white']} vs. {game_info['black']})\n"
        f"Total Games Analyzed: {len(session_data.get('processed_game_ids', []))}\n"
        f"Total Positions in Dataset: {len(session_data.get('all_ground_truth', {}))}\n"
        "\n--- LEADERBOARD ---\n"
        f"{leaderboard_text}\n"
        f"{'='*60}\n"
    )
    logging.info(update_message)

# --- SCORING FRAMEWORK IMPLEMENTATIONS ---

def calculate_move_scores_from_data(move_data, constants):
    """Calculates scores for a single move across all frameworks using pre-calculated data."""
    scores = {}
    
    best_move_eval = move_data['best_eval']
    played_move_eval = move_data['played_eval']
    if played_move_eval is not None and best_move_eval is not None:
        second_best_move_eval = move_data['second_best_eval']
        wp_best = 1 / (1 + np.exp(-constants['K1_WPL'] * best_move_eval))
        wp_played = 1 / (1 + np.exp(-constants['K1_WPL'] * played_move_eval))
        wp_loss = max(0, wp_best - wp_played)
        scores['RWPL'] = 100 * (1 - 2 * (min(wp_loss, 0.5) ** constants['K2_RWPL']))
        eval_gap = abs(best_move_eval - second_best_move_eval) if second_best_move_eval is not None else 0
        criticality = 1 / (1 + np.exp(-constants['K3_CAI'] * eval_gap))
        impact = wp_loss * (1 + constants['W_CAI'] * criticality)
        scores['CAI'] = 100 * (1 - 2 * (min(impact, 0.5) ** constants['K2_RWPL']))
        scores['Impact'] = scores['CAI']

    engine_specific_evals = move_data.get('complexity_evals')
    scores['Complexity'] = np.std(engine_specific_evals) if engine_specific_evals and len(engine_specific_evals) > 1 else 0.0

    played_move = move_data.get('played_move')
    oracle_moves = move_data.get('oracle_moves', [])
    if played_move and oracle_moves:
        for t_size in range(1, HITCOUNT_MAX_TEMPLATE_SIZE + 1):
            framework_name = f"HitCount_T{t_size}"
            score = 0.0
            template_slice = oracle_moves[:t_size]
            if played_move in template_slice:
                move_index = template_slice.index(played_move)
                if move_index == 0: score = 1.0
                elif move_index == 1: score = constants['HITCOUNT_SECOND_MOVE_WEIGHT']
            scores[framework_name] = score
            
    return scores

# --- CORE ANALYSIS SCRIPT ---

def get_standard_engine_options(csv_options_str, engine_info):
    options = {}
    if 'stockfish' in engine_info['name'].lower(): options["Hash"] = 256
    if csv_options_str:
        try: options.update(json.loads(csv_options_str))
        except json.JSONDecodeError: logging.warning(f"Could not decode JSON options: {csv_options_str}")
    return options

def print_progress_bar(iteration, total, prefix='', suffix='', length=50):
    if total == 0: total = 1
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: sys.stdout.write('\n')

def universal_worker(args):
    (fen, engine_info, limit_params, worker_mode) = args
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path'], stderr=subprocess.DEVNULL) as engine:
            options = get_standard_engine_options(engine_info.get('uci_options', '{}'), engine_info)
            engine.configure(options)
            board = chess.Board(fen)
            
            if worker_mode == 'oracle':
                info = engine.analyse(board, chess.engine.Limit(**limit_params), multipv=HITCOUNT_MAX_TEMPLATE_SIZE + 2)
                if not info: return None
                template = [{'move': m['pv'][0].uci(), 'eval': m['score'].pov(board.turn).score(mate_score=30000)} for m in info if m.get('score') and m.get('pv')]
                return (fen, template)

            elif worker_mode == 'model_simulation':
                info = engine.analyse(board, chess.engine.Limit(**limit_params), multipv=HITCOUNT_MAX_TEMPLATE_SIZE + 2)
                if not info or not info[0].get('pv'): return None
                played_move = info[0]['pv'][0]
                played_move_eval = info[0]['score'].pov(board.turn).score(mate_score=30000)
                complexity_evals = [m['score'].pov(board.turn).score(mate_score=30000) for m in info if m.get('score')]
                if played_move_eval is None: return None
                return (fen, engine_info['name'], {'move': played_move.uci(), 'eval': played_move_eval, 'complexity_evals': complexity_evals})
    except Exception as e:
        logging.error(f"Worker '{worker_mode}' for {engine_info['name']} on FEN '{fen}' failed: {e}")
    return None

def get_ground_truth_for_game(game, oracle_engine, pool):
    player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    candidate_nodes = [node for i, node in enumerate(game.mainline()) if node.board().turn == player_color and i > 20]
    if not candidate_nodes: return {}
    sample_size = min(len(candidate_nodes), POSITIONS_PER_GAME_TO_SAMPLE)
    sampled_indices = np.random.choice(len(candidate_nodes), sample_size, replace=False)
    fens_to_analyze = [candidate_nodes[i].board().fen() for i in sampled_indices]
    if not fens_to_analyze: return {}
    template, limit = {}, {'depth': ORACLE_ANALYSIS_DEPTH, 'time': ORACLE_TIMEOUT}
    tasks = [(fen, oracle_engine, limit, 'oracle') for fen in fens_to_analyze]
    print_progress_bar(0, len(tasks), prefix='Truth Template:')
    for i, result in enumerate(pool.imap_unordered(universal_worker, tasks)):
        if result: template[result[0]] = result[1]
        print_progress_bar(i + 1, len(tasks), prefix='Truth Template:')
    return template

def run_engine_simulations_for_game(ground_truth_template, engines, pool):
    sim_results, limit = defaultdict(dict), {'depth': MODEL_BUILD_DEPTH, 'time': MODEL_BUILD_TIMEOUT}
    tasks = [(fen, eng_info, limit, 'model_simulation') for eng_info in engines for fen in ground_truth_template]
    print_progress_bar(0, len(tasks), prefix='Simulations:')
    for i, result in enumerate(pool.imap_unordered(universal_worker, tasks)):
        if result: sim_results[result[0]][result[1]] = result[2]
        print_progress_bar(i + 1, len(tasks), prefix='Simulations:')
    return dict(sim_results)

def get_all_simulation_data(session_data):
    processed_data = []
    all_sim_results = session_data.get('all_sim_results', {})
    all_ground_truth = session_data.get('all_ground_truth', {})
    for fen, template in all_ground_truth.items():
        if not template or fen not in all_sim_results: continue
        oracle_moves = [m['move'] for m in template]
        for engine_name, move_data in all_sim_results[fen].items():
            processed_data.append({
                'engine_name': engine_name,
                'best_eval': template[0]['eval'],
                'second_best_eval': template[1]['eval'] if len(template) > 1 else None,
                'played_eval': move_data.get('eval'), 'played_move': move_data.get('move'),
                'oracle_moves': oracle_moves, 'complexity_evals': move_data.get('complexity_evals')
            })
    return processed_data

def evaluate_frameworks(all_processed_data, model_engines, constants, game_count, session_data, suppress_print=False):
    if not suppress_print:
        clear_console()
        logging.info(f"--- CUMULATIVE RESULTS AFTER {game_count} GAME(S) ---")
    
    engine_scores_by_framework = defaultdict(lambda: defaultdict(list))
    for move_data in all_processed_data:
        move_scores = calculate_move_scores_from_data(move_data, constants)
        for name, score in move_scores.items():
            engine_scores_by_framework[name][move_data['engine_name']].append(score)
    
    avg_scores_by_framework = {fw: {eng: np.mean(sc) for eng, sc in es.items() if sc} for fw, es in engine_scores_by_framework.items()}
    
    framework_models = {}
    for fw, scores_dict in avg_scores_by_framework.items():
        if not scores_dict: continue
        eng_data = [(eng['rating'], scores_dict[eng['name']]) for eng in model_engines if eng['name'] in scores_dict]
        if len(eng_data) < 3: continue
        ratings_data, scores_data = zip(*eng_data)
        ratings, scores = np.array(ratings_data).reshape(-1, 1), np.array(scores_data)
        model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(ratings, scores)
        r_squared = r2_score(scores, model.predict(ratings))
        framework_models[fw] = {'r_squared': r_squared, 'model': model, 'ratings': ratings, 'scores': scores}

    if not framework_models:
        if not suppress_print: logging.warning("Not enough data to build models yet.")
        return None, None, ""
        
    sorted_frameworks = sorted(framework_models.items(), key=lambda item: item[1]['r_squared'], reverse=True)
    leaderboard_text = "\n".join([f"  - {name:<15}: R² = {data['r_squared']:.6f}" for name, data in sorted_frameworks])
    if not suppress_print:
        full_leaderboard_text = f"{'='*50}\n{' '*15}FRAMEWORK LEADERBOARD\n{'='*50}\n{leaderboard_text}\n{'='*50}\n"
        print(full_leaderboard_text)
    
    return sorted_frameworks[0][0], framework_models, leaderboard_text

# --- MODE-SPECIFIC FUNCTIONS ---

def run_data_collection(pgn_path, oracle_engine, model_engines, session_data, start_time, max_duration_seconds):
    processed_game_ids = set(session_data.get('processed_game_ids', []))
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            while True:
                # Time limit check
                if time.time() - start_time > max_duration_seconds:
                    logging.info(f"Time limit of {max_duration_seconds / 3600:.2f} hours reached. Stopping data collection.")
                    break

                game_offset = pgn_file.tell()
                game = chess.pgn.read_game(pgn_file)
                if game is None: 
                    logging.info("Reached end of PGN file.")
                    break

                game_id = f"{game.headers.get('Date', '????.??.??')}_{game.headers.get('White', '?')}_{game.headers.get('Black', '?')}_{game_offset}"
                if game_id in processed_game_ids or PLAYER_TO_ANALYZE.lower() not in (game.headers.get("White", "?").lower(), game.headers.get("Black", "?").lower()):
                    continue

                white, black = game.headers.get('White', '?'), game.headers.get('Black', '?')
                logging.info(f"\n--- Analyzing Game {len(processed_game_ids) + 1}: {white} vs. {black} ---")
                
                game_truth = get_ground_truth_for_game(game, oracle_engine, pool)
                if not game_truth:
                    logging.warning("Could not generate ground truth for this game. Skipping.")
                    processed_game_ids.add(game_id)
                    session_data['processed_game_ids'] = list(processed_game_ids)
                    save_json_file(session_data, SESSION_FILE)
                    continue
                
                game_sims = run_engine_simulations_for_game(game_truth, model_engines, pool)
                session_data.setdefault('all_ground_truth', {}).update(game_truth)
                for fen, fen_sims in game_sims.items():
                    session_data.setdefault('all_sim_results', {}).setdefault(fen, {}).update(fen_sims)
                
                processed_game_ids.add(game_id)
                session_data['processed_game_ids'] = list(processed_game_ids)
                save_json_file(session_data, SESSION_FILE)
                logging.info(f"Saved progress for game {game_id} to session file.")
                
                yield session_data, {'id': game_id, 'white': white, 'black': black}

def create_final_report(best_constants, final_models, session_data, total_runtime_seconds):
    """Generates a formatted string with the final results of the run."""
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("          CHESS FRAMEWORK OPTIMIZER - FINAL REPORT")
    report_lines.append("="*60)
    report_lines.append(f"Run Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Runtime: {total_runtime_seconds / 3600:.2f} hours")
    report_lines.append(f"Total Games Analyzed: {len(session_data.get('processed_game_ids', []))}")
    report_lines.append(f"Total Positions in Dataset: {len(session_data.get('all_ground_truth', {}))}")
    report_lines.append("\n" + "-"*60)
    report_lines.append("          FINAL OPTIMIZED CONSTANTS")
    report_lines.append("-"*60)
    report_lines.append(json.dumps(best_constants, indent=4))
    report_lines.append("\n" + "-"*60)
    report_lines.append("          FINAL FRAMEWORK LEADERBOARD")
    report_lines.append("-"*60)
    
    if final_models:
        sorted_frameworks = sorted(final_models.items(), key=lambda item: item[1]['r_squared'], reverse=True)
        for name, data in sorted_frameworks:
            report_lines.append(f"  - {name:<15}: R² = {data['r_squared']:.6f}")
    else:
        report_lines.append("No models could be built with the collected data.")
        
    report_lines.append("="*60)
    return "\n".join(report_lines)

def run_analysis_mode(pgn_path, oracle_engine, model_engines, session_data, start_time, max_duration_seconds):
    constants = load_json_file(CONSTANTS_FILE, DEFAULT_CONSTANTS)
    logging.info(f"Starting analysis with constants: {constants}")
    
    for updated_session_data, game_info in run_data_collection(pgn_path, oracle_engine, model_engines, session_data, start_time, max_duration_seconds):
        game_count = len(updated_session_data.get('processed_game_ids', []))
        all_processed_data = get_all_simulation_data(updated_session_data)
        _, _, leaderboard_text = evaluate_frameworks(all_processed_data, model_engines, constants, game_count, updated_session_data)
        log_progress_update(leaderboard_text, game_info, updated_session_data)
    
    logging.info("--- Data collection complete. Generating final analysis. ---")
    all_processed_data = get_all_simulation_data(session_data)
    game_count = len(session_data.get('processed_game_ids', []))
    best_framework, final_models, _ = evaluate_frameworks(all_processed_data, model_engines, constants, game_count, session_data)
    
    if best_framework and final_models:
        bundle = final_models[best_framework]
        plt.figure(figsize=(12, 7))
        plt.scatter(bundle['ratings'].flatten(), bundle['scores'], alpha=0.7, label="Engine Performance")
        plot_x = np.linspace(bundle['ratings'].min(), bundle['ratings'].max(), 200).reshape(-1, 1)
        plt.plot(plot_x, bundle['model'].predict(plot_x), color='red', lw=2, label=f"Polyfit Model (R²={bundle['r_squared']:.4f})")
        plt.title(f"Final Result - Best Framework: '{best_framework}' Performance vs. Rating")
        plt.xlabel("Engine Elo Rating (from CSV)"); plt.ylabel(f"Average '{best_framework}' Score")
        plt.grid(True); plt.legend()
        graph_path = SESSION_FOLDER / f"final_analysis_graph.png"
        plt.savefig(graph_path); plt.close()
        logging.info(f"Final analysis graph saved to {graph_path}")

def run_optimization_mode(pgn_path, oracle_engine, model_engines, session_data, start_time, max_duration_seconds):
    logging.info("--- Starting Optimization Mode ---")
    logging.info("Step 1: Collecting simulation data from the training PGN...")
    
    # Data collection loop
    for updated_session_data, game_info in run_data_collection(pgn_path, oracle_engine, model_engines, session_data, start_time, max_duration_seconds):
        # Log progress during data collection
        game_count = len(updated_session_data.get('processed_game_ids', []))
        all_processed_data = get_all_simulation_data(updated_session_data)
        _, _, leaderboard_text = evaluate_frameworks(all_processed_data, model_engines, DEFAULT_CONSTANTS, game_count, updated_session_data, suppress_print=True)
        log_progress_update(leaderboard_text, game_info, updated_session_data)

    logging.info("Data collection complete.")
    all_processed_data = get_all_simulation_data(session_data)
    if not all_processed_data:
        logging.error("No simulation data collected. Cannot run optimization. Exiting."); return
    
    logging.info("\nStep 2: Optimizing all framework constants...")
    best_constants = load_json_file(CONSTANTS_FILE, DEFAULT_CONSTANTS)
    game_count = len(session_data.get('processed_game_ids', []))

    def get_framework_r2(constants_to_test, target_framework):
        _, models, _ = evaluate_frameworks(all_processed_data, model_engines, constants_to_test, game_count, session_data, suppress_print=True)
        if not models: return 0.0
        if target_framework == 'HitCount':
            hit_models = {name: data['r_squared'] for name, data in models.items() if name.startswith('HitCount_T')}
            return max(hit_models.values()) if hit_models else 0.0
        return models.get(target_framework, {}).get('r_squared', 0.0)

    for cycle in range(OPTIMIZATION_CYCLES):
        logging.info(f"\n--- Optimization Cycle {cycle + 1}/{OPTIMIZATION_CYCLES} ---")
        cai_r2 = get_framework_r2(best_constants, 'CAI')
        hitcount_r2 = get_framework_r2(best_constants, 'HitCount')
        logging.info(f"Starting cycle R² -> CAI: {cai_r2:.6f}, Best HitCount: {hitcount_r2:.6f}")

        for const_name, search_space in OPTIMIZATION_SEARCH_SPACE.items():
            target_framework = 'HitCount' if const_name == 'HITCOUNT_SECOND_MOVE_WEIGHT' else 'CAI'
            best_r2_for_const, best_value_for_const = -1, best_constants[const_name]
            for value in search_space:
                temp_constants = {**best_constants, const_name: value}
                current_r2 = get_framework_r2(temp_constants, target_framework)
                if current_r2 > best_r2_for_const:
                    best_r2_for_const, best_value_for_const = current_r2, value
            if best_constants[const_name] != best_value_for_const:
                logging.info(f"  -> Improved {const_name} for '{target_framework}': {best_constants[const_name]:.5f} -> {best_value_for_const:.5f} (New R²: {best_r2_for_const:.6f})")
                best_constants[const_name] = best_value_for_const
            else:
                logging.info(f"  -> No improvement for {const_name}. Stays at {best_constants[const_name]:.5f}")
        save_json_file(best_constants, CONSTANTS_FILE)

    logging.info("\n--- Optimization Complete ---")
    logging.info(f"Final optimized constants saved to {CONSTANTS_FILE}:")
    logging.info(json.dumps(best_constants, indent=4))

    # Generate and save the final report
    total_runtime_seconds = time.time() - start_time
    _, final_models, _ = evaluate_frameworks(all_processed_data, model_engines, best_constants, game_count, session_data, suppress_print=True)
    final_report_text = create_final_report(best_constants, final_models, session_data, total_runtime_seconds)
    
    logging.info("\n" + final_report_text)
    with open(FINAL_SUMMARY_FILE, 'w') as f:
        f.write(final_report_text)
    logging.info(f"Final summary report saved to {FINAL_SUMMARY_FILE}")


def main():
    start_time = time.time()
    max_duration_seconds = RUN_DURATION_HOURS * 3600

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
    
    all_engines.sort(key=lambda x: int(x['rating']), reverse=True)
    oracle_engine = all_engines[0]
    model_engines_full = all_engines[NUM_ORACLE_ENGINES:]
    indices = np.linspace(0, len(model_engines_full) - 1, MODEL_BUILD_ENGINES, dtype=int)
    model_engines = [model_engines_full[i] for i in indices]
    logging.info(f"Oracle: {oracle_engine['name']}. Model builders: {len(model_engines)} engines.")
    logging.info(f"Script will run for a maximum of {RUN_DURATION_HOURS} hours.")

    clear_console()
    print("="*60 + f"\n Chess Framework Optimizer & Analyzer (v8 - Timed Run)\n" + "="*60)
    print("This script is configured for a timed optimization run.")
    print("It will collect data for up to 8.5 hours, then automatically")
    print("calculate the best constants and generate a final report.")
    
    pgn_path_str = input("Enter the full path to your PGN file: ").strip().strip('"')
    pgn_path = Path(pgn_path_str)
    if not pgn_path.is_file():
        logging.error(f"PGN file not found at {pgn_path}"); return

    session_data = load_json_file(SESSION_FILE)
    script_version = 'v8'
    if session_data.get('pgn_file') != str(pgn_path) or session_data.get('script_version') != script_version:
        logging.warning("New PGN/script version detected or no session found. Starting a new session.")
        session_data = {'pgn_file': str(pgn_path), 'script_version': script_version}

    # Directly run optimization mode for the automated run
    run_optimization_mode(pgn_path, oracle_engine, model_engines, session_data, start_time, max_duration_seconds)
    
    logging.info("--- Script finished ---")


if __name__ == "__main__":
    if sys.platform.startswith('darwin') or sys.platform.startswith('win'):
        multiprocessing.set_start_method('spawn', force=True)
    main()
