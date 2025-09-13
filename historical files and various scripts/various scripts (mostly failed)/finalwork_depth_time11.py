# chess_analyzer_multi_engine_v35_adaptive_r3.py
# A script to analyze PGN chess games and estimate a specific player's rating.
#
# KEY CHANGES:
#   1. UNCONDITIONAL CHECKPOINT: For each game, a checkpoint phase is run on a
#      small, sparse set of engines (20) to determine the best analysis path.
#   2. DUAL-STRATEGY & MULTI-TEMPLATE TESTING: The checkpoint phase tests BOTH
#      depth-based and time-based analysis. For each strategy, it calculates
#      the model's R-squared (R²) value for multiple template sizes (e.g.,
#      considering the top 1, 2, or 3 oracle moves as a 'hit').
#   3. OPTIMAL PATH SELECTION: The script compares all R² values from the
#      checkpoint and selects the absolute best combination of analysis
#      strategy (depth vs. time) and template size for that specific game.
#   4. ADAPTIVE FINAL ANALYSIS: The final, more detailed analysis is performed
#      using the winning parameters on a larger set of engines (30 for depth-
#      based, 15 for time-based) to ensure both accuracy and efficiency.
#   5. COMPREHENSIVE LOGGING: All checkpoint results and the final decision
#      are logged, providing a clear audit trail.

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
MODEL_SWITCH_CHECKPOINT = 20 # Number of engines to test before checking R² to switch strategy
NUM_DEPTH_ANALYSIS_POINTS = 60 # Number of engines for the primary depth-based analysis
NUM_SPARSE_ANALYSIS_POINTS = 30 # Number of engines for the fallback time-based analysis

# --- CCRL-INSPIRED ENGINE SETTINGS ---
HASH_SIZE_MB = 256
TABLEBASE_PATH = r"" # Set to your Syzygy path, or leave blank to disable

# --- HIT-BASED ANALYSIS CONTROLS ---
TEMPLATE_MOVE_COUNT = 3

# --- PRIMARY (DEPTH-PRIORITY) ANALYSIS CONTROLS ---
ORACLE_ANALYSIS_DEPTH = 20
ORACLE_TIMEOUT = 300 # (in seconds)

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

def get_sparse_engines(full_engine_list, num_points):
    """Selects a subset of engines with evenly spaced ratings."""
    if len(full_engine_list) <= num_points:
        return full_engine_list

    sorted_engines = sorted(full_engine_list, key=lambda x: x['rating'])
    ratings = np.array([eng['rating'] for eng in sorted_engines])
    min_rating, max_rating = 1350, 3150
    target_ratings = np.linspace(min_rating, max_rating, num_points)
    
    selected_indices = set()
    for target in target_ratings:
        closest_index = np.argmin(np.abs(ratings - target))
        selected_indices.add(closest_index)
        
    sparse_engine_list = [sorted_engines[i] for i in sorted(list(selected_indices))]
    logging.info(f"Selected a sparse set of {len(sparse_engine_list)} engines for analysis.")
    return sparse_engine_list

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

def estimate_rating_from_quality(model, quality_score):
    """Estimates Elo by finding the rating that maps to the given quality score."""
    search_ratings = np.arange(1350, 3151, 1)
    predicted_qualities = model.predict(search_ratings.reshape(-1, 1))
    return int(search_ratings[np.argmin(np.abs(predicted_qualities - quality_score))])

# --- ISOLATED WORKER FUNCTIONS ---

def universal_worker(args):
    """A universal worker for all analysis types."""
    (fen, player_move, engine_info, limit_params, ground_truth_template, worker_mode, template_size) = args

    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            options = get_standard_engine_options(engine_info.get('uci_options', '{}'))
            engine.configure(options)
            board = chess.Board(fen)

            if worker_mode == 'oracle':
                info = engine.analyse(board, chess.engine.Limit(**limit_params), multipv=TEMPLATE_MOVE_COUNT)
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
                        raw_score_white = move_info['score'].white().score(mate_score=30000)
                        if raw_score_white is None: continue
                        template.append({'move': move.uci(), 'score': quality_score, 'raw_score': raw_score_white})
                return template

            elif worker_mode == 'model_simulation':
                result = engine.play(board, chess.engine.Limit(**limit_params))
                return result.move.uci() if result.move else None

            elif worker_mode == 'player_analysis':
                if not ground_truth_template.get(fen): return None
                
                template_moves = [entry['move'] for entry in ground_truth_template[fen][:template_size]]
                if player_move.uci() in template_moves:
                    for entry in ground_truth_template[fen]:
                        if entry['move'] == player_move.uci():
                            return entry['score']
                
                best_move_score_white_pov = ground_truth_template[fen][0]['raw_score']
                info = engine.analyse(board, chess.engine.Limit(**limit_params), root_moves=[player_move])
                if not info: return None
                played_move_score_white_pov = info['score'].white().score(mate_score=30000)
                if played_move_score_white_pov is None: return None
                cpl = max(0, (best_move_score_white_pov - played_move_score_white_pov) if board.turn == chess.WHITE else (played_move_score_white_pov - best_move_score_white_pov))
                return calculate_punishment_score(cpl)

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
    tasks = [(fen, None, oracle_engines[0], limit, None, 'oracle', None) for fen in fens_to_analyze]
    print_progress_bar(0, len(tasks), prefix=f'Truth Template ({oracle_engines[0]["name"]}):', suffix='Complete', length=50)
    with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
        for i, result in enumerate(pool.imap_unordered(universal_worker, tasks)):
            if result:
                template[tasks[i][0]] = result
            print_progress_bar(i + 1, len(tasks), prefix=f'Truth Template ({oracle_engines[0]["name"]}):', suffix='Complete', length=50)

    session_data['in_progress_game']['ground_truth_template'] = template
    save_session(session_data, pgn_path)
    return template

def run_engine_simulations(ground_truth_template, engines_to_process, session_data, pgn_path, analysis_params, is_checkpoint=False):
    """Runs engine simulations and stores the moves played."""
    sim_results_key = f"model_engine_moves_{analysis_params['name']}"
    if sim_results_key in session_data['in_progress_game']:
        # Check if results for the current set of engines already exist
        sim_data = session_data['in_progress_game'][sim_results_key]
        first_fen = next(iter(ground_truth_template))
        engine_names_to_process = {e['name'] for e in engines_to_process}
        if first_fen in sim_data and engine_names_to_process.issubset(sim_data[first_fen].keys()):
             logging.info(f"Found existing simulation results for '{analysis_params['name']}'. Skipping simulation.")
             return

    logging.info(f"Running engine simulations with params: {analysis_params['name']}...")
    limit = {k: v for k, v in analysis_params.items() if k in ['depth', 'time'] and v is not None}
    
    all_tasks = []
    for fen in ground_truth_template.keys():
        for engine_info in engines_to_process:
            all_tasks.append((fen, None, engine_info, limit, None, 'model_simulation', None))

    prefix = "Checkpoint Sims:" if is_checkpoint else "Building Sims:"
    print_progress_bar(0, len(all_tasks), prefix=prefix, suffix='Complete', length=50)
    
    simulation_results = defaultdict(dict)
    with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
        for i, move_uci in enumerate(pool.imap_unordered(universal_worker, all_tasks)):
            task_fen = all_tasks[i][0]
            task_engine_name = all_tasks[i][2]['name']
            if move_uci:
                simulation_results[task_fen][task_engine_name] = move_uci
            print_progress_bar(i + 1, len(all_tasks), prefix=prefix, suffix='Complete', length=50)

    session_data['in_progress_game'][sim_results_key] = dict(simulation_results)
    save_session(session_data, pgn_path)

def calculate_scores_from_sims(simulation_results, ground_truth_template, template_size):
    """Calculates average scores for each engine based on a given template size."""
    engine_scores = defaultdict(list)
    for fen, moves_by_engine in simulation_results.items():
        if fen not in ground_truth_template: continue
        
        template_moves = {entry['move']: entry['score'] for entry in ground_truth_template[fen][:template_size]}

        for engine_name, move_uci in moves_by_engine.items():
            if move_uci in template_moves:
                engine_scores[engine_name].append(template_moves[move_uci])
            else:
                engine_scores[engine_name].append(0.0) # Simplified punishment for a miss
    
    return {name: np.mean(scores) if scores else 0.0 for name, scores in engine_scores.items()}

def build_model_from_scores(engine_scores, engines_to_process):
    """Builds a regression model from pre-calculated scores."""
    ratings_data, quality_scores_data = [], []
    for eng_info in engines_to_process:
        if eng_info['name'] in engine_scores:
            ratings_data.append(eng_info['rating'])
            quality_scores_data.append(engine_scores[eng_info['name']])

    if len(ratings_data) < 3: return None
    
    ratings = np.array(ratings_data).reshape(-1, 1)
    quality_scores = np.array(quality_scores_data)
    
    linear_model = LinearRegression().fit(ratings, quality_scores)
    poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(ratings, quality_scores)
    
    test_ratings = np.arange(ratings.min(), ratings.max(), 10).reshape(-1, 1)
    poly_predictions = poly_model.predict(test_ratings)
    is_monotonic = np.all(np.diff(poly_predictions) >= 0)

    linear_r2 = r2_score(quality_scores, linear_model.predict(ratings))
    poly_r2 = r2_score(quality_scores, poly_model.predict(ratings)) if is_monotonic else -1.0

    best_model, best_r2, best_model_name = (poly_model, poly_r2, 'Polynomial') if poly_r2 > linear_r2 else (linear_model, linear_r2, 'Linear')

    return {'model': best_model, 'r_squared': best_r2, 'ratings': ratings, 'quality_scores': quality_scores, 'model_type': best_model_name}

def analyze_player(game, ground_truth_template, oracle_engine, analysis_params, template_size):
    """Analyzes the player's moves using a specific template size."""
    logging.info(f"Analyzing player moves with template size {template_size}...")
    player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    tasks = []
    board = game.board()
    limit = {k:v for k,v in analysis_params.items() if k in ['depth', 'time'] and v is not None}
    
    for move in game.mainline_moves():
        if board.turn == player_color:
            tasks.append((board.fen(), move, oracle_engine, limit, ground_truth_template, 'player_analysis', template_size))
        board.push(move)

    if not tasks: return 0.0, 0
    scores = []
    print_progress_bar(0, len(tasks), prefix=f'Player Analysis (T={template_size}):', suffix='Complete', length=50)
    with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
        for i, score in enumerate(pool.imap_unordered(universal_worker, tasks)):
            if score is not None: scores.append(score)
            print_progress_bar(i + 1, len(tasks), prefix=f'Player Analysis (T={template_size}):', suffix='Complete', length=50)
    return (np.mean(scores) if scores else 0.0), len(scores)

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

            # --- NEW ADAPTIVE CHECKPOINT LOGIC ---
            chosen_strategy_params = analysis_strategies['depth_first']
            chosen_template_size = TEMPLATE_MOVE_COUNT
            engines_for_final_run = get_sparse_engines(model_engines, NUM_DEPTH_ANALYSIS_POINTS)
            decision_log = ""
            
            if len(model_engines) >= MODEL_SWITCH_CHECKPOINT:
                checkpoint_engines = get_sparse_engines(model_engines, MODEL_SWITCH_CHECKPOINT)
                checkpoint_results = []

                # Unconditionally test both strategies
                for strategy_name, params in analysis_strategies.items():
                    logging.info(f"--- Running Checkpoint: {strategy_name.replace('_', ' ').title()} Strategy ---")
                    run_engine_simulations(ground_truth_template, checkpoint_engines, session_data, pgn_path, params, is_checkpoint=True)
                    sim_results = session_data['in_progress_game'][f"model_engine_moves_{params['name']}"]
                    
                    for t_size in range(1, TEMPLATE_MOVE_COUNT + 1):
                        scores = calculate_scores_from_sims(sim_results, ground_truth_template, t_size)
                        model_bundle = build_model_from_scores(scores, checkpoint_engines)
                        if model_bundle:
                            checkpoint_results.append({'r2': model_bundle['r_squared'], 'strategy': strategy_name, 'template_size': t_size})
                
                # Determine the best overall strategy and template size from all tests
                best_overall_result = max(checkpoint_results, key=lambda x: x['r2']) if checkpoint_results else None
                
                if best_overall_result:
                    chosen_strategy_name = best_overall_result['strategy']
                    chosen_strategy_params = analysis_strategies[chosen_strategy_name]
                    chosen_template_size = best_overall_result['template_size']
                    
                    if chosen_strategy_name == 'time_first':
                        engines_for_final_run = get_sparse_engines(model_engines, NUM_SPARSE_ANALYSIS_POINTS)
                    
                    decision_log = f"Best checkpoint combo: '{chosen_strategy_name}' with template size {chosen_template_size} (R²={best_overall_result['r2']:.4f})."
                    session_data['in_progress_game']['checkpoint_results'] = checkpoint_results
                else:
                    decision_log = "Could not determine best strategy from checkpoint. Defaulting to depth-first."

            else:
                decision_log = "Not enough model engines for a checkpoint test. Defaulting to depth-first."

            logging.info(decision_log)
            logging.info(f"--- Proceeding with full analysis using '{chosen_strategy_params['name']}' (T={chosen_template_size}) on {len(engines_for_final_run)} engines. ---")
            
            run_engine_simulations(ground_truth_template, engines_for_final_run, session_data, pgn_path, chosen_strategy_params)
            final_sim_results = session_data['in_progress_game'][f"model_engine_moves_{chosen_strategy_params['name']}"]
            final_scores = calculate_scores_from_sims(final_sim_results, ground_truth_template, chosen_template_size)
            final_model_bundle = build_model_from_scores(final_scores, engines_for_final_run)
            
            if not final_model_bundle:
                logging.error("Final analysis run failed. Skipping game.")
                session_data['games_to_process_indices'].pop(0); save_session(session_data, pgn_path); continue

            avg_score, moves_counted = analyze_player(game, ground_truth_template, oracle_engines[0], chosen_strategy_params, chosen_template_size)
            if moves_counted == 0:
                logging.warning("Player analysis yielded no moves. Skipping game.")
                session_data['games_to_process_indices'].pop(0); save_session(session_data, pgn_path); continue

            est_rating = estimate_rating_from_quality(final_model_bundle['model'], avg_score)

            plt.figure(figsize=(10, 6))
            plt.scatter(final_model_bundle['ratings'].flatten(), final_model_bundle['quality_scores'], alpha=0.7, label="Engine Performance")
            plot_x = np.linspace(final_model_bundle['ratings'].min(), final_model_bundle['ratings'].max(), 200).reshape(-1, 1)
            plt.plot(plot_x, final_model_bundle['model'].predict(plot_x), color='red', lw=2, label=f"{final_model_bundle['model_type']} Model (R²={final_model_bundle['r_squared']:.4f})")
            plt.axhline(y=avg_score, color='g', linestyle='--', label=f"Player Score ({avg_score:.4f}) -> {est_rating} Elo")
            plt.title(f"Game {game_num}: {white} vs. {black} (Analysis: {chosen_strategy_params['name']}, T={chosen_template_size})")
            plt.xlabel("Engine Elo Rating (from CSV)"); plt.ylabel("Weighted Score"); plt.grid(True); plt.legend()
            graph_path = SESSION_FOLDER / f"performance_graph_game_{game_num}.png"
            plt.savefig(graph_path); plt.close()

            game_summary = f"Game {game_num}: {white} vs. {black} -> Est. Rating: {est_rating} (Model: {final_model_bundle['model_type']}, R²={final_model_bundle['r_squared']:.3f}, Moves: {moves_counted})"
            logging.info(game_summary)
            with open(report_file_path, 'a', encoding='utf-8') as report_file:
                report_file.write(game_summary + "\n")
                report_file.write(f"  > Decision: {decision_log}\n")

            session_data.setdefault('completed_games_data', []).append({
                'game_num': game_num, 'white': white, 'black': black, 'r_squared': final_model_bundle['r_squared'],
                'graph_path': str(graph_path), 'model_type': final_model_bundle['model_type'], 'avg_score': avg_score,
                'moves': moves_counted, 'estimated_rating': est_rating,
                'analysis_decision': decision_log,
                'checkpoint_data': session_data['in_progress_game'].get('checkpoint_results', [])
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
