import chess
import chess.pgn
import chess.engine
import os
import json
import random
from tqdm import tqdm
import pandas as pd

# ==============================================================================
# --- Configuration ---
# ==============================================================================
PGN_FILE = "chessgames_august2025.pgn"
ORACLE_ENGINE_PATH = r"C:\Users\desja\Documents\my_programming\chess_analysis\engines\stockfish\stockfish-windows-x86-64-sse41-popcnt.exe"
ORACLE_CACHE_FILE = "oracle_cache_top3.json" 
GRANULAR_LOG_FILE = "granular_analysis_log_top3.csv"
ENGINES_CSV_PATH = "real_engines.csv"

PLAYER_NAME = "Desjardins373"
MIN_POSITIONS_TO_SAMPLE = 500
ORACLE_MAX_DEPTH = 30 
# Using a more balanced ply count for sampling
MIN_PLY_FOR_SAMPLING = 40 # This is 20 moves for each player

# ==============================================================================
# --- Utility Functions ---
# ==============================================================================

def load_json_cache(file_path):
    """Loads a JSON cache from a file if it exists."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"[WARNING] Could not decode JSON from {file_path}. Starting fresh.")
                return {}
    return {}

def save_json_cache(cache, file_path):
    """Saves a cache to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(cache, f, indent=4)

def sample_player_positions_and_moves(pgn_file, player_name, num_positions_target):
    """
    Randomly samples positions from a player's games where it was their turn to move.
    Returns a dictionary mapping {fen: player_move_uci}.
    """
    print(f"[SAMPLING] Sampling positions from {player_name}'s games in {pgn_file}...")
    player_moves_map = {}
    if not os.path.exists(pgn_file):
        print(f"[ERROR] PGN file not found: {pgn_file}")
        return player_moves_map

    candidate_positions = []
    with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                player_color = None
                if game.headers.get("White", "?").lower() == player_name.lower():
                    player_color = chess.WHITE
                elif game.headers.get("Black", "?").lower() == player_name.lower():
                    player_color = chess.BLACK
                else:
                    continue

                moves = list(game.mainline_moves())
                if len(moves) <= MIN_PLY_FOR_SAMPLING:
                    continue

                board = game.board()
                for i, move in enumerate(moves):
                    # Check if ply is past the minimum and it's the player's turn
                    if i >= MIN_PLY_FOR_SAMPLING and board.turn == player_color:
                        candidate_positions.append({"fen": board.fen(), "player_move": move.uci()})
                    board.push(move)
            except (ValueError, IndexError) as e:
                print(f"[WARNING] Skipping a malformed game in PGN: {e}")
                continue
    
    if len(candidate_positions) > num_positions_target:
        sampled_data = random.sample(candidate_positions, num_positions_target)
    else:
        sampled_data = candidate_positions
        print(f"[WARNING] Found only {len(candidate_positions)} candidate positions, which is less than the target of {num_positions_target}.")

    for item in sampled_data:
        player_moves_map[item["fen"]] = item["player_move"]
        
    print(f"[SAMPLING] Sampled {len(player_moves_map)} unique positions for analysis.")
    return player_moves_map

def generate_oracle_moves(positions_to_analyze, oracle_cache):
    """Generates oracle's top 3 best moves, saving progress continuously."""
    print(f"[ORACLE] Generating top 3 oracle moves with Stockfish "
          f"(max depth {ORACLE_MAX_DEPTH}, 2 threads).")

    try:
        with chess.engine.SimpleEngine.popen_uci(ORACLE_ENGINE_PATH) as oracle:
            oracle.configure({"Threads": 2})
            
            # This loop is now safe to interrupt with Ctrl+C
            for fen in tqdm(positions_to_analyze, desc="Generating oracle moves"):
                if fen in oracle_cache:
                    continue
                board = chess.Board(fen)
                try:
                    analysis_results = oracle.analyse(
                        board,
                        chess.engine.Limit(depth=ORACLE_MAX_DEPTH),
                        multipv=3 
                    )
                    
                    top_moves = [
                        info["pv"][0].uci() 
                        for info in analysis_results 
                        if "pv" in info and info["pv"]
                    ]

                    if top_moves:
                        oracle_cache[fen] = top_moves
                    else:
                        print(f"[WARNING] Oracle found no principal variation for FEN: {fen}")

                except chess.engine.EngineTerminatedError as e:
                    print(f"[ERROR] Oracle engine terminated unexpectedly: {e}")
                    # Break the loop but still save progress in the finally block
                    break 
    
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during oracle generation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # This block is GUARANTEED to run, even if the script is stopped or crashes.
        print("\n[ORACLE] Saving oracle cache progress...")
        save_json_cache(oracle_cache, ORACLE_CACHE_FILE)
        print(f"[ORACLE] Save complete. Cache now has {len(oracle_cache)} positions.")

    return oracle_cache

def run_evaluations(oracle_cache, player_moves_map):
    """
    Runs player and engines against the oracle positions, logging score and move rank.
    """
    print("\n[EVALUATION] Beginning player and engine evaluations...")

    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Engines CSV file not found: {ENGINES_CSV_PATH}")
        return

    if os.path.exists(GRANULAR_LOG_FILE):
        log_df = pd.read_csv(GRANULAR_LOG_FILE)
        completed = set(zip(log_df["engine_name"], log_df["fen"]))
    else:
        log_df = pd.DataFrame(columns=["engine_name", "fen", "score", "move_rank"])
        completed = set()

    contestants = [{"name": PLAYER_NAME, "type": "player"}]
    for _, row in engines_df.iterrows():
        contestants.append({"name": row["engine_name"], "type": "engine", "data": row.to_dict()})

    positions_to_evaluate = list(player_moves_map.keys())

    for contestant in contestants:
        contestant_name = contestant["name"]
        print(f"\n--- Analyzing with {contestant_name} ---")
        
        new_results = []
        
        def calculate_score_and_rank(played_move, oracle_top_moves):
            move_rank = -1
            if played_move in oracle_top_moves:
                move_rank = oracle_top_moves.index(played_move)
            score = 1 if move_rank == 0 else 0
            return score, move_rank

        if contestant["type"] == "player":
            for fen in tqdm(positions_to_evaluate, desc=f"Evaluating {contestant_name}"):
                if (contestant_name, fen) in completed or fen not in oracle_cache:
                    continue
                
                player_move = player_moves_map.get(fen)
                oracle_top_moves = oracle_cache.get(fen)
                
                if player_move and oracle_top_moves:
                    score, move_rank = calculate_score_and_rank(player_move, oracle_top_moves)
                    new_results.append({
                        "engine_name": contestant_name, 
                        "fen": fen, 
                        "score": score, 
                        "move_rank": move_rank
                    })
        
        elif contestant["type"] == "engine":
            engine_data = contestant["data"]
            engine_path = engine_data["path"]
            uci_options_str = str(engine_data.get("uci_options", "{}"))

            try:
                uci_options = json.loads(uci_options_str)
                limit_args = {
                    "time": uci_options.get("time"),
                    "depth": uci_options.get("Depth"),
                    "nodes": uci_options.get("Nodes")
                }
                limit_args = {k: v for k, v in limit_args.items() if v is not None}
                
                if not limit_args:
                    limit_args['time'] = 0.5
                
                limit = chess.engine.Limit(**limit_args)

                with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
                    engine.configure({"Threads": 1 if "maia" in contestant_name.lower() else 2})

                    for fen in tqdm(positions_to_evaluate, desc=f"Evaluating {contestant_name}"):
                        if (contestant_name, fen) in completed or fen not in oracle_cache:
                            continue

                        oracle_top_moves = oracle_cache.get(fen)
                        if not oracle_top_moves: continue

                        try:
                            board = chess.Board(fen)
                            info = engine.analyse(board, limit)
                            engine_best_move = info.get("pv")[0].uci() if info.get("pv") else None
                            
                            if engine_best_move:
                                score, move_rank = calculate_score_and_rank(engine_best_move, oracle_top_moves)
                                new_results.append({
                                    "engine_name": contestant_name, 
                                    "fen": fen, 
                                    "score": score, 
                                    "move_rank": move_rank
                                })
                        except Exception as e:
                            print(f"[ERROR] {contestant_name} failed on FEN {fen}: {e}")
            
            except Exception as e:
                print(f"[ERROR] A critical error occurred while running {contestant_name}: {e}")

        if new_results:
            log_df = pd.concat([log_df, pd.DataFrame(new_results)], ignore_index=True)
            log_df.to_csv(GRANULAR_LOG_FILE, index=False)
            print(f"[SAVE] Saved {len(new_results)} new results for {contestant_name}.")

    print(f"\n[DONE] All evaluations saved to {GRANULAR_LOG_FILE}")

def main():
    """Main function to orchestrate the rating analysis."""
    print("--- Starting Rating Estimation Script (Top 3 Moves) ---")

    player_moves_map = sample_player_positions_and_moves(PGN_FILE, PLAYER_NAME, MIN_POSITIONS_TO_SAMPLE)
    if not player_moves_map:
        print("[ERROR] No positions were sampled from the PGN file. Exiting.")
        return

    oracle_cache = load_json_cache(ORACLE_CACHE_FILE)
    print(f"[CACHE] Oracle cache currently has {len(oracle_cache)} positions.")
    
    new_positions_for_oracle = [p for p in player_moves_map.keys() if p not in oracle_cache]

    if new_positions_for_oracle:
        print(f"[ORACLE] {len(new_positions_for_oracle)} new positions need oracle analysis.")
        # The generate_oracle_moves function now handles its own saving internally
        generate_oracle_moves(new_positions_for_oracle, oracle_cache)
    else:
        print("[ORACLE] Oracle cache is up-to-date for all sampled positions.")

    run_evaluations(oracle_cache, player_moves_map)

if __name__ == "__main__":
    main()
