import chess
import chess.pgn
import pandas as pd
import sys
import os
import json
from tqdm import tqdm
import random

# ==============================================================================
# --- Configuration (Should match your original script) ---
# ==============================================================================

PLAYER_NAME_IN_PGN = "Desjardins373"
PLAYER_PGN_PATH = "chessgames_august2025.pgn"
GRANULAR_LOG_PATH = "granular_analysis_log.csv"
ORACLE_CACHE_PATH = "oracle_cache.json"

CHOSEN_METHOD = {
    "weights": [1.0, 0.5, 0.25]
}

# --- Analysis Control ---
START_MOVE = 10
POSITIONS_PER_GAME = 5

# ==============================================================================
# --- Helper Functions (from your main script) ---
# ==============================================================================

def get_positions_from_pgn(pgn_path, player_name, positions_per_game, start_move):
    """
    Extracts a list of FENs and the player's move from a PGN file.
    Randomly samples positions from each game, skipping non-standard variants.
    """
    if not os.path.exists(pgn_path):
        print(f"Error: PGN file not found at '{pgn_path}'.", file=sys.stderr)
        return []
        
    all_selected_positions = []
    print(f"Processing PGN: {os.path.basename(pgn_path)} for player: {player_name}")
    with open(pgn_path, 'r', errors='ignore') as pgn:
        while True:
            try:
                game = chess.pgn.read_game(pgn)
                if game is None: break

                # --- FIX: Skip Chess960 and other variants ---
                if "Variant" in game.headers and game.headers["Variant"] != "Standard":
                    continue
                
                white_player = game.headers.get("White", "")
                black_player = game.headers.get("Black", "")
                if player_name not in white_player and player_name not in black_player:
                    continue

                board = game.board()
                is_player_white = player_name in white_player
                
                eligible_positions_in_game = []
                for move in game.mainline_moves():
                    is_player_turn = (is_player_white and board.turn == chess.WHITE) or \
                                     (not is_player_white and board.turn == chess.BLACK)
                    
                    if is_player_turn and board.fullmove_number >= start_move:
                        eligible_positions_in_game.append({"fen": board.fen(), "actual_move": move.uci()})
                    
                    board.push(move)
                
                if positions_per_game is not None and len(eligible_positions_in_game) > 0:
                    if len(eligible_positions_in_game) > positions_per_game:
                        sampled_positions = random.sample(eligible_positions_in_game, positions_per_game)
                        all_selected_positions.extend(sampled_positions)
                    else:
                        all_selected_positions.extend(eligible_positions_in_game)
                elif positions_per_game is None:
                    all_selected_positions.extend(eligible_positions_in_game)

            except Exception as e:
                print(f"Skipping a game due to a parsing error: {e}", file=sys.stderr)
                continue
    return all_selected_positions


def get_move_score(move_to_check, oracle_moves, weights):
    """Calculates the score for a move based on the oracle's ranking."""
    if not oracle_moves: return 0.0
    try:
        index = oracle_moves.index(move_to_check)
        if index < len(weights):
            return weights[index]
    except ValueError:
        return 0.0
    return 0.0

# ==============================================================================
# --- Main Logic ---
# ==============================================================================

def main():
    """Calculates player scores and appends them to the granular log."""
    print("--- Add Player Scores to Log Script ---")

    # --- 1. Load Caches and PGN Data ---
    if not os.path.exists(ORACLE_CACHE_PATH):
        print(f"Error: Oracle cache not found at '{ORACLE_CACHE_PATH}'. Exiting."); return
    with open(ORACLE_CACHE_PATH, 'r') as f:
        oracle_cache = json.load(f)
    print(f"Loaded {len(oracle_cache)} positions from Oracle cache.")

    all_player_positions = get_positions_from_pgn(PLAYER_PGN_PATH, PLAYER_NAME_IN_PGN, POSITIONS_PER_GAME, START_MOVE)
    if not all_player_positions:
        print("No standard positions found for the player in the PGN. Exiting."); return
    print(f"Found {len(all_player_positions)} positions played by {PLAYER_NAME_IN_PGN}.")

    # --- 2. Check for Already Completed Work ---
    completed_items = set()
    if os.path.exists(GRANULAR_LOG_PATH):
        try:
            log_df = pd.read_csv(GRANULAR_LOG_PATH)
            if not log_df.empty:
                player_log = log_df[log_df['engine_name'] == 'player']
                completed_items = set(player_log['fen'])
                print(f"Found {len(completed_items)} player scores already in the log.")
        except (pd.errors.EmptyDataError, KeyError):
             print(f"Log file '{GRANULAR_LOG_PATH}' is empty or malformed.")

    # --- 3. Calculate and Append Player Scores ---
    player_rows_to_add = []
    for pos in tqdm(all_player_positions, desc="Scoring Player Moves"):
        if pos['fen'] not in completed_items:
            score = get_move_score(pos['actual_move'], oracle_cache.get(pos['fen'], []), CHOSEN_METHOD['weights'])
            player_rows_to_add.append({'fen': pos['fen'], 'engine_name': 'player', 'score': score})

    if player_rows_to_add:
        print(f"\nAdding {len(player_rows_to_add)} new player scores to the log.")
        player_rows_df = pd.DataFrame(player_rows_to_add)
        is_new_file = not os.path.exists(GRANULAR_LOG_PATH) or os.path.getsize(GRANULAR_LOG_PATH) == 0
        player_rows_df.to_csv(GRANULAR_LOG_PATH, mode='a', header=is_new_file, index=False)
    else:
        print("\nPlayer scores are already up-to-date in the log.")

    print("\n--- Script Finished ---")
    print("You can now run your optimization and graphing script successfully.")

if __name__ == "__main__":
    main()
