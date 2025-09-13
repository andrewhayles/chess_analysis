# chess_analyzer_multi_engine_v33_weighted_template.py
# A script to analyze PGN chess games and estimate a specific player's rating.
#
# KEY CHANGE (from v32):
#   1. ADVANCED WEIGHTED SCORING: Implements a more sophisticated "hit-based"
#      scoring system as requested.
#   2. NORMALIZED TEMPLATE: The oracle generates a template of the top 3 moves,
#      and their scores are normalized to a 0.0-1.0 quality scale.
#   3. WEIGHTED HITS: A "hit" on a template move is no longer a flat 1.0 score.
#      Instead, it receives the normalized quality score for that specific move,
#      rewarding better choices within the template more highly.

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
from functools import partial

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(__file__).resolve().parent
ENGINES_CSV_PATH = PROJECT_FOLDER / "real_engines.csv"
SESSION_FOLDER = PROJECT_FOLDER / "chess_analysis_session"
SESSION_FOLDER.mkdir(exist_ok=True)

PLAYER_TO_ANALYZE = "Desjardins373"
NUM_ANALYSIS_CORES = 2
NUM_ORACLE_ENGINES = 1

# --- CCRL-INSPIRED ENGINE SETTINGS ---
HASH_SIZE_MB = 256
TABLEBASE_PATH = r"" # Set to your Syzygy path, or leave blank to disable

# --- HIT-BASED ANALYSIS CONTROLS ---
TEMPLATE_MOVE_COUNT = 3

# --- DEPTH-PRIORITY ANALYSIS CONTROLS ---
ORACLE_ANALYSIS_DEPTH = 20
ORACLE_TIMEOUT = 600 # (in seconds)

MODEL_BUILD_DEPTH = 12
MODEL_BUILD_TIMEOUT = 6 # (in seconds)

PLAYER_ANALYSIS_DEPTH = 12
PLAYER_ANALYSIS_TIMEOUT = 6 # (in seconds)

# --- SCRIPT SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(SESSION_FOLDER / 'analysis_log.txt'), logging.StreamHandler(sys.stdout)]
)
progress_logger = logging.getLogger('progress_logger')
# ... (logging setup remains the same)

# --- HELPER FUNCTIONS ---

def get_standard_engine_options(csv_options_str):
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
    if total == 0: total = 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total: sys.stdout.write('\n'); sys.stdout.flush()

def calculate_punishment_score(centipawn_loss):
    """Calculates a score for a 'miss' based on CPL."""
    if not isinstance(centipawn_loss, (int, float)) or centipawn_loss is None: return 0.0
    if centipawn_loss <= 0: return 0.90 # Cap miss score below any possible hit
    if centipawn_loss >= 200: return 0.0
    return 0.90 * (1 - (centipawn_loss / 200.0))

# --- ISOLATED WORKER FUNCTIONS ---

def oracle_template_worker(args):
    """Oracle worker to get a template of the top N moves with normalized scores."""
    fen, engine_info, depth, timeout, multipv_count = args
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            options = get_standard_engine_options(engine_info.get('uci_options', '{}'))
            engine.configure(options)
            info = engine.analyse(chess.Board(fen), chess.engine.Limit(depth=depth, time=timeout), multipv=multipv_count)
            
            template = []
            if not info: return None

            # First, get the score of the absolute best move to use as a baseline
            best_score = info[0]['score'].white().score(mate_score=30000)
            if best_score is None: return None

            for move_info in info:
                if 'pv' in move_info and 'score' in move_info:
                    move = move_info['pv'][0]
                    current_score = move_info['score'].white().score(mate_score=30000)
                    if current_score is None: continue
                    
                    # Calculate CPL relative to the best move
                    cpl = max(0, best_score - current_score)
                    # Use the main quality function to get a normalized score for this template move
                    quality_score = calculate_move_quality(cpl)

                    template.append({'move': move.uci(), 'score': quality_score})
            return template
    except Exception as e:
        logging.error(f"Oracle template worker for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
    return None

def weighted_hit_worker(args):
    """A universal worker for the new weighted hit-based scoring methodology."""
    fen, player_move, depth, timeout, engine_info, ground_truth_template = args
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_info['path']) as engine:
            options = get_standard_engine_options(engine_info.get('uci_options', '{}'))
            engine.configure(options)
            
            board = chess.Board(fen)
            
            if player_move is None:
                result = engine.play(board, chess.engine.Limit(depth=depth, time=timeout))
                if not result.move: return None
                move_played = result.move
            else:
                move_played = player_move

            # Check if the move is a "hit"
            for template_entry in ground_truth_template[fen]:
                if move_played.uci() == template_entry['move']:
                    return template_entry['score'] # Return the pre-calculated weighted score

            # If it's a "miss", calculate the punishment score
            board.push(move_played)
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=timeout), info=chess.engine.INFO_SCORE)
            if isinstance(info, list): info = info[0]

            if 'score' in info:
                player_to_move = chess.Board(fen).turn
                played_move_score = info['score'].pov(player_to_move).score(mate_score=30000)
                
                # Get the raw score of the absolute best move from the original analysis
                # We need to re-calculate the best score in the POV of the player to move
                best_move_raw_score_white_pov = chess.engine.PovScore(chess.Cp(int(ground_truth_template[fen][0]['raw_score'])), chess.WHITE).pov(player_to_move).score(mate_score=30000)

                cpl = max(0, best_move_raw_score_white_pov - played_move_score)
                return calculate_punishment_score(cpl)

    except Exception as e:
        logging.error(f"Weighted hit worker for {engine_info['name']} failed on FEN {fen}: {repr(e)}")
    return None

# --- MAIN ANALYSIS FUNCTIONS ---

def get_ground_truth_template(game, oracle_engines, session_data, pgn_path):
    in_progress_data = session_data.get('in_progress_game', {})
    if in_progress_data.get('offset') == game.offset and 'ground_truth_template' in in_progress_data:
        logging.info("Loading pre-calculated ground truth template from session file.")
        return in_progress_data['ground_truth_template']

    logging.info("Generating ground truth template...")
    player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    fens_to_analyze = [node.board().fen() for node in game.mainline() if node.board().turn == player_color]
    if not fens_to_analyze: return {}

    template = defaultdict(list)
    for engine_info in oracle_engines:
        tasks = [(fen, engine_info, ORACLE_ANALYSIS_DEPTH, ORACLE_TIMEOUT, TEMPLATE_MOVE_COUNT) for fen in fens_to_analyze]
        print_progress_bar(0, len(tasks), prefix=f'Truth Template ({engine_info["name"]}):', suffix='Complete', length=50)
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            for i, result in enumerate(pool.imap_unordered(oracle_template_worker, tasks)):
                if result: template[tasks[i][0]] = result
                print_progress_bar(i + 1, len(tasks), prefix=f'Truth Template ({engine_info["name"]}):', suffix='Complete', length=50)

    session_data['in_progress_game']['ground_truth_template'] = template
    save_session(session_data, pgn_path)
    return template

def build_model_weighted_hit(ground_truth_template, model_engines, session_data, pgn_path):
    logging.info(f"Building weighted-hit performance model...")
    engine_scores = session_data.get('in_progress_game', {}).get('model_engine_results', {})
    for i, engine_info in enumerate(model_engines):
        if engine_info['name'] in engine_scores: continue
        logging.info(f"Processing model engine {i+1}/{len(model_engines)}: {engine_info['name']}")
        tasks = [(fen, None, MODEL_BUILD_DEPTH, MODEL_BUILD_TIMEOUT, engine_info, ground_truth_template) for fen in ground_truth_template]
        scores_list = []
        print_progress_bar(0, len(tasks), prefix=f'Building ({engine_info["name"]}):', suffix='Complete', length=50)
        with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
            for j, score in enumerate(pool.imap_unordered(weighted_hit_worker, tasks)):
                if score is not None: scores_list.append(score)
                print_progress_bar(j + 1, len(tasks), prefix=f'Building ({engine_info["name"]}):', suffix='Complete', length=50)
        if scores_list:
            engine_scores[engine_info['name']] = np.mean(scores_list)
            session_data['in_progress_game']['model_engine_results'] = engine_scores
            save_session(session_data, pgn_path)

    ratings_data, quality_scores_data = [], []
    for eng_info in model_engines:
        if eng_info['name'] in engine_scores:
            ratings_data.append(eng_info['rating'])
            quality_scores_data.append(engine_scores[eng_info['name']])

    if len(ratings_data) < 3: return None, None, None, None, None, None
    ratings = np.array(ratings_data).reshape(-1, 1)
    quality_scores = np.array(quality_scores_data)
    models = {
        'Linear': {'model': LinearRegression().fit(ratings, quality_scores)},
        'Polynomial': {'model': make_pipeline(PolynomialFeatures(degree=3), LinearRegression()).fit(ratings, quality_scores)}
    }
    for name in models: models[name]['r2'] = models[name]['model'].score(ratings, quality_scores)
    best_model_name = max(models, key=lambda name: models[name]['r2'])
    logging.info(f"Model selected: {best_model_name}, R²={models[best_model_name]['r2']:.4f}")
    return models[best_model_name]['model'], models[best_model_name]['r2'], ratings, quality_scores, best_model_name, engine_scores

def analyze_player_weighted_hit(game, ground_truth_template, analysis_engine_info):
    logging.info(f"Analyzing player moves (weighted-hit)...")
    player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else chess.BLACK
    tasks = []
    board = game.board()
    for move in game.mainline_moves():
        if board.turn == player_color:
            tasks.append((board.fen(), move, PLAYER_ANALYSIS_DEPTH, PLAYER_ANALYSIS_TIMEOUT, analysis_engine_info, ground_truth_template))
        board.push(move)
    if not tasks: return 0.0, 0
    
    scores = []
    print_progress_bar(0, len(tasks), prefix='Player Analysis:', suffix='Complete', length=50)
    with multiprocessing.Pool(processes=NUM_ANALYSIS_CORES) as pool:
        for i, score in enumerate(pool.imap_unordered(weighted_hit_worker, tasks)):
            if score is not None: scores.append(score)
            print_progress_bar(i + 1, len(tasks), prefix='Player Analysis:', suffix='Complete', length=50)
    return (np.mean(scores) if scores else 0.0), len(scores)

# ... (estimate_rating, save_session, load_session remain the same) ...
def estimate_rating_from_quality(model, quality_score):
    search_ratings = np.arange(1350, 3151, 1)
    predicted_qualities = model.predict(search_ratings.reshape(-1, 1))
    return int(search_ratings[np.argmin(np.abs(predicted_qualities - quality_score))])

def save_session(session_data, pgn_path):
    with open(SESSION_FOLDER / pgn_path.with_suffix('.session.json').name, 'w') as f:
        json.dump(session_data, f, indent=4)
    logging.info(f"Session progress saved to {SESSION_FOLDER / pgn_path.with_suffix('.session.json').name}")

def load_session(pgn_path):
    session_file = SESSION_FOLDER / pgn_path.with_suffix('.session.json').name
    if session_file.exists():
        try:
            with open(session_file, 'r') as f: return json.load(f)
        except json.JSONDecodeError: return None
    return None

# --- MAIN EXECUTION LOGIC ---
def main():
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
    logging.info(f"Loaded {len(oracle_engines)} oracle engines and {len(model_engines)} model-building engines.")

    pgn_path = Path(input("Enter the full path to your PGN file: "))
    if not pgn_path.is_file(): logging.error(f"PGN file not found at {pgn_path}"); return

    session_data = load_session(pgn_path)
    if not session_data:
        session_data = {'pgn_file': str(pgn_path), 'completed_games_data': [], 'in_progress_game': {}}
        with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
            offsets = [pgn_file.tell() for h in iter(lambda: chess.pgn.read_headers(pgn_file), None) if PLAYER_TO_ANALYZE.lower() in (h.get("White", "?").lower(), h.get("Black", "?").lower())]
            session_data['games_to_process_indices'] = offsets
        save_session(session_data, pgn_path)

    logging.info(f"Found {len(session_data['games_to_process_indices'])} games remaining to analyze.")
    progress_logger.info(f"Chess Analysis Report for {PLAYER_TO_ANALYZE} from '{pgn_path.name}'")
    progress_logger.info("-" * 50)

    continuous_mode = False
    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn_file:
        while session_data['games_to_process_indices']:
            offset = session_data['games_to_process_indices'][0]
            pgn_file.seek(offset)
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                session_data['games_to_process_indices'].pop(0); continue
            game.offset = offset
            if session_data.get('in_progress_game', {}).get('offset') != offset:
                session_data['in_progress_game'] = {'offset': offset}

            white, black = game.headers.get('White', '?'), game.headers.get('Black', '?')
            game_num = len(session_data['completed_games_data']) + 1
            logging.info(f"--- Starting Analysis for Game {game_num}: {white} vs. {black} ---")
            
            ground_truth_template = get_ground_truth_template(game, oracle_engines, session_data, pgn_path)
            if not ground_truth_template:
                session_data['games_to_process_indices'].pop(0); save_session(session_data, pgn_path); continue

            model_results = build_model_weighted_hit(ground_truth_template, model_engines, session_data, pgn_path)
            if model_results[0] is None:
                session_data['games_to_process_indices'].pop(0); save_session(session_data, pgn_path); continue
            
            model, r_squared, ratings, quality_scores, model_type, raw_model_results = model_results
            
            avg_score, moves_counted = analyze_player_weighted_hit(game, ground_truth_template, oracle_engines[0])
            if moves_counted == 0:
                session_data['games_to_process_indices'].pop(0); save_session(session_data, pgn_path); continue

            est_rating = estimate_rating_from_quality(model, avg_score)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(ratings.flatten(), quality_scores, alpha=0.7, label="Engine Performance")
            plot_x = np.linspace(ratings.min(), ratings.max(), 200).reshape(-1, 1)
            plt.plot(plot_x, model.predict(plot_x), color='red', lw=2, label=f"{model_type} Model (R²={r_squared:.4f})")
            plt.axhline(y=avg_score, color='g', linestyle='--', label=f"Player Score ({avg_score:.4f}) -> {est_rating} Elo")
            plt.title(f"Game {game_num}: {white} vs. {black}"); plt.xlabel("Engine Elo Rating (from CSV)"); plt.ylabel("Proportion of Hits / Weighted Score"); plt.grid(True); plt.legend()
            graph_path = SESSION_FOLDER / f"performance_graph_game_{game_num}.png"
            plt.savefig(graph_path); plt.close()
            
            progress_logger.info(f"Game {game_num}: {white} vs. {black} -> Est. Rating: {est_rating} (Model: {model_type}, R²={r_squared:.3f}, Moves: {moves_counted})")

            session_data['completed_games_data'].append({
                'game_num': game_num, 'white': white, 'black': black, 'r_squared': r_squared,
                'graph_path': str(graph_path), 'model_type': model_type, 'avg_score': avg_score,
                'moves': moves_counted, 'estimated_rating': est_rating, 'model_results_raw': raw_model_results
            })
            session_data['games_to_process_indices'].pop(0)
            session_data['in_progress_game'] = {}
            save_session(session_data, pgn_path)
            logging.info(f"--- Finished Analysis for Game {game_num}. Estimated rating: {est_rating}. ---")

            if not continuous_mode and session_data['games_to_process_indices']:
                user_input = input("Continue with next game? (y/n/c for yes/no/continuous): ").lower().strip()
                if user_input == 'n': break
                if user_input == 'c': continuous_mode = True

    logging.info("All games analyzed.")
    completed_games = session_data.get('completed_games_data', [])
    if completed_games:
        final_avg_rating = int(np.mean([g['estimated_rating'] for g in completed_games]))
        summary = f"\nFinal Average Estimated Rating: {final_avg_rating} Elo across {len(completed_games)} games."
        progress_logger.info("-" * 50); progress_logger.info(summary)
        logging.info(summary)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
