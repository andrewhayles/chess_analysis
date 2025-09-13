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

def get_all_player_positions(pgn_file, player_name):
    """
    Scans the entire PGN and returns a map of all positions and moves for the player.
    This is used for the one-time cache upgrade.
    """
    print(f"[PGN SCAN] Scanning {pgn_file} for all of {player_name}'s moves...")
    player_moves_map = {}
    if not os.path.exists(pgn_file):
        print(f"[ERROR] PGN file not found: {pgn_file}")
        return {}

    with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
        game_count = 0
        while True:
            try:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                game_count += 1
                player_color = None
                if game.headers.get("White", "?").lower() == player_name.lower():
                    player_color = chess.WHITE
                elif game.headers.get("Black", "?").lower() == player_name.lower():
                    player_color = chess.BLACK
                else:
                    continue

                board = game.board()
                for move in game.mainline_moves():
                    if board.turn == player_color:
                        player_moves_map[board.fen()] = move.uci()
                    board.push(move)
            except (ValueError, IndexError) as e:
                print(f"[WARNING] Skipping a malformed game in PGN: {e}")
                continue
    print(f"[PGN SCAN] Found {len(player_moves_map)} total moves across {game_count} games.")
    return player_moves_map


def generate_and_save_oracle_moves(positions_to_analyze, oracle_cache, player_moves_map):
    """
    Generates oracle moves and saves them in the new, robust format.
    Format: {fen: {"oracle_moves": [...], "player_move": "..."}}
    """
    print(f"[ORACLE] Generating top 3 oracle moves with Stockfish (max depth {ORACLE_MAX_DEPTH}, 2 threads).")
    try:
        with chess.engine.SimpleEngine.popen_uci(ORACLE_ENGINE_PATH) as oracle:
            oracle.configure({"Threads": 2})
            for fen in tqdm(positions_to_analyze, desc="Generating oracle moves"):
                board = chess.Board(fen)
                try:
                    analysis_results = oracle.analyse(board, chess.engine.Limit(depth=ORACLE_MAX_DEPTH), multipv=3)
                    top_moves = [info["pv"][0].uci() for info in analysis_results if "pv" in info and info["pv"]]
                    if top_moves:
                        player_move = player_moves_map.get(fen)
                        if player_move:
                            oracle_cache[fen] = {"oracle_moves": top_moves, "player_move": player_move}
                    else:
                        print(f"[WARNING] Oracle found no principal variation for FEN: {fen}")
                except chess.engine.EngineTerminatedError as e:
                    print(f"[ERROR] Oracle engine terminated unexpectedly: {e}")
                    break
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during oracle generation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[ORACLE] Saving oracle cache progress...")
        save_json_cache(oracle_cache, ORACLE_CACHE_FILE)
        print(f"[ORACLE] Save complete. Cache now has {len(oracle_cache)} positions.")

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

    log_df = pd.read_csv(GRANULAR_LOG_FILE) if os.path.exists(GRANULAR_LOG_FILE) else pd.DataFrame(columns=["engine_name", "fen", "score", "move_rank"])
    completed = set(zip(log_df["engine_name"], log_df["fen"]))

    contestants = [{"name": PLAYER_NAME, "type": "player"}]
    for _, row in engines_df.iterrows():
        contestants.append({"name": row["engine_name"], "type": "engine", "data": row.to_dict()})

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

        positions_to_evaluate = [fen for fen in player_moves_map.keys() if (contestant_name, fen) not in completed]

        if contestant["type"] == "player":
            for fen in tqdm(positions_to_evaluate, desc=f"Evaluating {contestant_name}"):
                cache_entry = oracle_cache.get(fen)
                if not cache_entry: continue
                
                player_move = player_moves_map.get(fen)
                oracle_top_moves = cache_entry.get("oracle_moves")

                if player_move and oracle_top_moves:
                    score, move_rank = calculate_score_and_rank(player_move, oracle_top_moves)
                    new_results.append({"engine_name": contestant_name, "fen": fen, "score": score, "move_rank": move_rank})
        
        elif contestant["type"] == "engine":
            engine_data = contestant["data"]
            try:
                uci_options = json.loads(str(engine_data.get("uci_options", "{}")))
                limit_args = {k: v for k, v in {"time": uci_options.get("time"), "depth": uci_options.get("Depth"), "nodes": uci_options.get("Nodes")}.items() if v is not None}
                if not limit_args: limit_args['time'] = 0.5
                limit = chess.engine.Limit(**limit_args)

                with chess.engine.SimpleEngine.popen_uci(engine_data["path"]) as engine:
                    engine.configure({"Threads": 1 if "maia" in contestant_name.lower() else 2})
                    for fen in tqdm(positions_to_evaluate, desc=f"Evaluating {contestant_name}"):
                        cache_entry = oracle_cache.get(fen)
                        if not cache_entry: continue
                        oracle_top_moves = cache_entry.get("oracle_moves")
                        if not oracle_top_moves: continue
                        try:
                            board = chess.Board(fen)
                            info = engine.analyse(board, limit)
                            engine_best_move = info.get("pv")[0].uci() if info.get("pv") else None
                            if engine_best_move:
                                score, move_rank = calculate_score_and_rank(engine_best_move, oracle_top_moves)
                                new_results.append({"engine_name": contestant_name, "fen": fen, "score": score, "move_rank": move_rank})
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
    
    oracle_cache = load_json_cache(ORACLE_CACHE_FILE)
    print(f"[CACHE] Oracle cache currently has {len(oracle_cache)} positions.")

    # Check if the cache is in the new format {fen: {"oracle_moves": [...], "player_move": "..."}}
    is_v2_format = False
    if oracle_cache:
        first_item = next(iter(oracle_cache.values()))
        if isinstance(first_item, dict) and "oracle_moves" in first_item and "player_move" in first_item:
            is_v2_format = True

    player_moves_map = {}
    
    # --- The Final, Correct Logic ---
    if len(oracle_cache) >= MIN_POSITIONS_TO_SAMPLE and is_v2_format:
        # Case 1: Everything is perfect. The cache is full and in the new format.
        print("[CACHE] V2 format cache is sufficient. Skipping PGN sampling and analysis.")
        player_moves_map = {fen: data["player_move"] for fen, data in oracle_cache.items()}
    
    else:
        # Case 2: The cache is either old, incomplete, or both.
        if not is_v2_format and oracle_cache:
            # Subcase 2a: The cache exists but is in the old format. We need to upgrade it.
            print("[CACHE UPGRADE] Old format detected. Performing fast, one-time upgrade without re-analysis...")
            all_player_moves = get_all_player_positions(PGN_FILE, PLAYER_NAME)
            
            upgraded_cache = {}
            upgraded_count = 0
            for fen, oracle_moves_list in oracle_cache.items():
                if fen in all_player_moves:
                    upgraded_cache[fen] = {
                        "oracle_moves": oracle_moves_list,
                        "player_move": all_player_moves[fen]
                    }
                    upgraded_count += 1
            
            save_json_cache(upgraded_cache, ORACLE_CACHE_FILE)
            oracle_cache = upgraded_cache # Use the upgraded cache for the rest of this run
            player_moves_map = {fen: data["player_move"] for fen, data in oracle_cache.items()}
            print(f"[CACHE UPGRADE] Upgrade complete. {upgraded_count} positions updated and saved. Future runs will be immediate.")

        else:
            # Subcase 2b: The cache is empty or insufficient. We need to build it from scratch.
            print("[CACHE] Cache is empty or insufficient. Building from PGN...")
            player_moves_map = sample_player_positions_and_moves(PGN_FILE, PLAYER_NAME, MIN_POSITIONS_TO_SAMPLE)
            if not player_moves_map:
                print("[ERROR] No positions were sampled from the PGN file. Exiting.")
                return
            
            positions_to_analyze = [p for p in player_moves_map.keys() if p not in oracle_cache]
            if positions_to_analyze:
                print(f"[ORACLE] {len(positions_to_analyze)} new positions need analysis.")
                generate_and_save_oracle_moves(positions_to_analyze, oracle_cache, player_moves_map)
            else:
                 print("[ORACLE] No new positions to analyze.")


    if not player_moves_map:
        # This can happen if the upgrade fails to find matches.
        print("[ERROR] Could not load player moves from cache or PGN. Exiting.")
        return
        
    run_evaluations(oracle_cache, player_moves_map)

if __name__ == "__main__":
    main()
