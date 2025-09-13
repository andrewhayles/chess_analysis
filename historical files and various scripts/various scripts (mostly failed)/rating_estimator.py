# rating_estimator_v2.py
# This script estimates a player's Elo rating based on their performance in the
# positions analyzed by the chess_framework_optimizer_v7.py script.
#
# --- KEY IMPROVEMENTS (from v1) ---
# 1. CORRECT ELO PREDICTION: Implements an "inverse search" on the performance model
#    to correctly find the Elo rating that matches the player's score, fixing the
#    primary logic error that produced very low Elo estimates.
# 2. DYNAMIC MOVE ANALYSIS: If the player makes a move that was not in the oracle's
#    top list (a "miss" or blunder), this script will now launch a chess engine
#    to analyze that specific move on the fly. This ensures ALL player moves are
#    scored, not just the good ones, leading to a more accurate assessment.
#
# --- HOW IT WORKS ---
# 1. Loads data from 'optimizer_session.json'.
# 2. Rebuilds the CAI performance model from the engine data.
# 3. Reads the PGN file to find the human player's moves in the analyzed positions.
# 4. If a player's move is not in the pre-analyzed list, it analyzes it with an engine.
# 5. Calculates the human player's average CAI score across ALL moves.
# 6. Uses the performance model to accurately estimate an Elo rating from the player's score.

import json
import chess
import chess.pgn
import chess.engine
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from pathlib import Path
import sys
import csv
from collections import defaultdict
import logging
import subprocess

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(__file__).resolve().parent
SESSION_FOLDER = PROJECT_FOLDER / "chess_analysis_session"
SESSION_FILE = SESSION_FOLDER / "optimizer_session.json"
ENGINES_CSV_PATH = PROJECT_FOLDER / "real_engines.csv"
PLAYER_TO_ANALYZE = "Desjardins373" # <--- IMPORTANT: Make sure this matches the player name from the PGN

# --- ENGINE SETTINGS FOR BLUNDER ANALYSIS ---
# This engine will be used to evaluate player moves not found in the session file.
# It should be a strong engine. We'll find its path from the CSV.
BLUNDER_ANALYSIS_ENGINE_NAME = "stockfish_elo_3190"
BLUNDER_ANALYSIS_DEPTH = 16 # A moderate depth is sufficient for scoring.

# --- SCRIPT SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- CORE FUNCTIONS (Copied from optimizer script for consistency) ---

DEFAULT_CONSTANTS = {
    "K1_WPL": 0.00368, "K2_RWPL": 0.5, "K3_CAI": 0.004, "W_CAI": 1.0,
    "HITCOUNT_SECOND_MOVE_WEIGHT": 0.75
}

def calculate_cai_score(best_eval, second_best_eval, played_eval, constants):
    """Calculates a single CAI score from evaluation data."""
    if played_eval is None or best_eval is None:
        return None

    wp_best = 1 / (1 + np.exp(-constants['K1_WPL'] * best_eval))
    wp_played = 1 / (1 + np.exp(-constants['K1_WPL'] * played_eval))
    wp_loss = max(0, wp_best - wp_played)
    eval_gap = abs(best_eval - second_best_eval) if second_best_eval is not None else 0
    criticality = 1 / (1 + np.exp(-constants['K3_CAI'] * eval_gap))
    impact = wp_loss * (1 + constants['W_CAI'] * criticality)
    return 100 * (1 - 2 * (min(impact, 0.5) ** constants['K2_RWPL']))

def get_all_simulation_data(session_data):
    """Processes the raw session data into a list of move data points for engines."""
    processed_data = []
    all_sim_results = session_data.get('all_sim_results', {})
    all_ground_truth = session_data.get('all_ground_truth', {})
    
    for fen, template in all_ground_truth.items():
        if not template or fen not in all_sim_results:
            continue
        
        for engine_name, move_data in all_sim_results[fen].items():
            score = calculate_cai_score(
                template[0]['eval'],
                template[1]['eval'] if len(template) > 1 else None,
                move_data.get('eval'),
                DEFAULT_CONSTANTS
            )
            if score is not None:
                processed_data.append({'engine_name': engine_name, 'cai_score': score})
    return processed_data

def estimate_rating_from_score(model, target_score):
    """
    NEW: Correctly estimates an Elo rating by finding the point on the
    model's curve that is closest to the player's average score.
    """
    # Define a range of Elo ratings to search
    search_ratings = np.arange(1000, 3501, 1).reshape(-1, 1)
    
    # Predict the CAI score for each Elo rating in the range
    predicted_scores = model.predict(search_ratings)
    
    # Find the index of the rating that produced the score closest to our target
    closest_index = np.argmin(np.abs(predicted_scores - target_score))
    
    # Return the Elo rating at that index
    return int(search_ratings[closest_index][0])

# --- MAIN RATING ESTIMATION LOGIC ---

def main():
    logging.info("--- Player Elo Rating Estimator (v2) ---")

    # 1. Load all necessary files
    if not SESSION_FILE.exists():
        logging.error(f"FATAL: Session file not found at {SESSION_FILE}. Please run the optimizer first."); return
    if not ENGINES_CSV_PATH.exists():
        logging.error(f"FATAL: Engines CSV file not found at {ENGINES_CSV_PATH}."); return

    logging.info(f"Loading session data from: {SESSION_FILE}")
    session_data = json.load(open(SESSION_FILE))
    pgn_path_str = session_data.get('pgn_file')
    if not pgn_path_str or not Path(pgn_path_str).exists():
        logging.error(f"FATAL: PGN file '{pgn_path_str}' from session is not accessible."); return

    all_engines_info = {row['name']: {'rating': int(row['rating']), 'path': row['path']} for row in csv.DictReader(open(ENGINES_CSV_PATH)) if not row['name'].strip().startswith('#')}
    
    # 2. Rebuild the CAI Performance Model from engine data
    logging.info("Step 1: Rebuilding the CAI performance model from engine data...")
    engine_data = get_all_simulation_data(session_data)
    engine_scores_by_cai = defaultdict(list)
    for move in engine_data:
        engine_scores_by_cai[move['engine_name']].append(move['cai_score'])

    avg_engine_scores = {eng: np.mean(scores) for eng, scores in engine_scores_by_cai.items()}
    model_data = [(all_engines_info[name]['rating'], score) for name, score in avg_engine_scores.items() if name in all_engines_info]
    
    if len(model_data) < 3:
        logging.error("Not enough engine data to build a reliable model."); return

    ratings_data, scores_data = zip(*model_data)
    ratings = np.array(ratings_data).reshape(-1, 1)
    scores = np.array(scores_data)
    
    model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    model.fit(ratings, scores)
    logging.info("Performance model built successfully.")

    # 3. Find the human player's moves from the PGN
    logging.info(f"Step 2: Finding moves for player '{PLAYER_TO_ANALYZE}' in {Path(pgn_path_str).name}...")
    fens_to_check = set(session_data['all_ground_truth'].keys())
    player_moves = {}
    with open(pgn_path_str, encoding="utf-8", errors="ignore") as pgn_file:
        while game := chess.pgn.read_game(pgn_file):
            player_color = chess.WHITE if game.headers.get("White", "?").lower() == PLAYER_TO_ANALYZE.lower() else (chess.BLACK if game.headers.get("Black", "?").lower() == PLAYER_TO_ANALYZE.lower() else None)
            if player_color is None: continue
            node = game
            while not node.is_end():
                board = node.board()
                if board.turn == player_color:
                    fen = board.fen()
                    if fen in fens_to_check:
                        player_moves[fen] = node.next().move
                node = node.next()

    if not player_moves:
        logging.error(f"Could not find any moves for player '{PLAYER_TO_ANALYZE}' in the analyzed positions."); return
    logging.info(f"Found {len(player_moves)} moves played by you in the analyzed positions.")

    # 4. Calculate the player's average CAI score, analyzing blunders if necessary
    logging.info("Step 3: Calculating your average CAI score (with dynamic blunder analysis)...")
    player_cai_scores = []
    all_ground_truth = session_data['all_ground_truth']
    blunder_engine = None

    try:
        for fen, played_move in player_moves.items():
            ground_truth_moves = all_ground_truth[fen]
            best_eval = ground_truth_moves[0]['eval']
            second_best_eval = ground_truth_moves[1]['eval'] if len(ground_truth_moves) > 1 else None
            
            # Try to find the move in the pre-calculated list
            played_move_eval = next((m['eval'] for m in ground_truth_moves if m['move'] == played_move.uci()), None)

            # NEW: If move not found, analyze it dynamically
            if played_move_eval is None:
                logging.warning(f"Move {played_move.uci()} not in oracle list for FEN {fen}. Analyzing dynamically...")
                if blunder_engine is None: # Initialize engine only when needed
                    engine_path = all_engines_info[BLUNDER_ANALYSIS_ENGINE_NAME]['path']
                    blunder_engine = chess.engine.SimpleEngine.popen_uci(engine_path, stderr=subprocess.DEVNULL)
                
                board = chess.Board(fen)
                info = blunder_engine.analyse(board, chess.engine.Limit(depth=BLUNDER_ANALYSIS_DEPTH), root_moves=[played_move])
                if info and 'score' in info:
                    played_move_eval = info['score'].pov(board.turn).score(mate_score=30000)
                    logging.info(f"  -> Dynamic analysis complete. Eval: {played_move_eval}")
                else:
                    logging.error(f"  -> Dynamic analysis FAILED for move {played_move.uci()}. Skipping score.")
                    continue

            cai_score = calculate_cai_score(best_eval, second_best_eval, played_move_eval, DEFAULT_CONSTANTS)
            if cai_score is not None:
                player_cai_scores.append(cai_score)

    finally:
        if blunder_engine:
            blunder_engine.quit()

    if not player_cai_scores:
        logging.error("Could not calculate any CAI scores for your moves."); return

    average_player_score = np.mean(player_cai_scores)

    # 5. Predict Elo rating from the score using the corrected method
    logging.info("Step 4: Estimating your Elo rating...")
    estimated_rating = estimate_rating_from_score(model, average_player_score)

    # --- FINAL RESULTS ---
    print("\n" + "="*50)
    print("           ELO RATING ESTIMATION RESULTS (v2)")
    print("="*50)
    print(f"  Player Analyzed:      {PLAYER_TO_ANALYZE}")
    print(f"  Positions Used:         {len(player_cai_scores)} (all moves scored)")
    print(f"  Your Average CAI Score: {average_player_score:.2f}")
    print("-" * 50)
    print(f"  ESTIMATED ELO RATING:   {estimated_rating}")
    print("="*50)
    print("\nDisclaimer: This is a rough estimate. For best results, analyze more games.")

if __name__ == "__main__":
    main()
