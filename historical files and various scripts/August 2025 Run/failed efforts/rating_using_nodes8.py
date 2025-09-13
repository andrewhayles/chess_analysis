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
ORACLE_CACHE_FILE = "oracle_cache.json"
GRANULAR_LOG_FILE = "granular_analysis_log.csv"
ENGINES_CSV_PATH = "real_engines.csv"

PLAYER_NAME = "Desjardins373"
TARGET_POSITIONS = 500  # minimum positions needed before engine testing
ORACLE_TIME_LIMIT = 600  # failsafe time limit (seconds)
ORACLE_MAX_DEPTH = 22    # max depth for oracle
POSITIONS_PER_GAME = 5   # fallback sampling amount if cache < TARGET_POSITIONS

# ==============================================================================
# --- Utility Functions ---
# ==============================================================================
def load_oracle_cache():
    """Loads the oracle cache from a JSON file if it exists."""
    if os.path.exists(ORACLE_CACHE_FILE):
        with open(ORACLE_CACHE_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"[WARNING] Could not decode JSON from {ORACLE_CACHE_FILE}. Starting with an empty cache.")
                return {}
    return {}

def save_oracle_cache(cache):
    """Saves the given cache to a JSON file."""
    with open(ORACLE_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

def sample_positions_from_pgn(pgn_file, per_game=5):
    """Randomly samples legal positions from each game after move 10."""
    positions = []
    if not os.path.exists(pgn_file):
        print(f"[ERROR] PGN file not found at: {pgn_file}")
        return positions

    with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                # Check if the specified player is in the game
                if game.headers.get("White", "?").lower() != PLAYER_NAME.lower() and \
                   game.headers.get("Black", "?").lower() != PLAYER_NAME.lower():
                    continue

                moves = list(game.mainline_moves())
                if len(moves) <= 10:
                    continue

                # Pick ply indices (not moves) after the 10th move (ply 20)
                # We sample from indices 19 onwards (to get positions *after* move 10)
                start_ply = 20
                if len(moves) > start_ply:
                    sample_indices = random.sample(
                        range(start_ply, len(moves)), 
                        min(per_game, len(moves) - start_ply)
                    )

                    board = game.board()
                    for i, move in enumerate(moves):
                        board.push(move)
                        if i in sample_indices:
                            positions.append(board.fen())
            except (ValueError, IndexError) as e:
                print(f"[WARNING] Skipping a malformed game in PGN: {e}")
                continue
    return positions


def generate_oracle_moves(positions, oracle_cache):
    """Generates oracle moves using a strong engine with depth/time limits."""
    print(f"[ORACLE] Generating oracle moves with Stockfish "
          f"(max depth {ORACLE_MAX_DEPTH}, max {ORACLE_TIME_LIMIT}s each, 2 threads).")

    try:
        with chess.engine.SimpleEngine.popen_uci(ORACLE_ENGINE_PATH) as oracle:
            oracle.configure({"Threads": 2})
            for fen in tqdm(positions, desc="Generating oracle moves"):
                if fen in oracle_cache:
                    continue
                board = chess.Board(fen)
                try:
                    # Limit both time and depth for the oracle analysis
                    info = oracle.analyse(
                        board,
                        chess.engine.Limit(time=ORACLE_TIME_LIMIT, depth=ORACLE_MAX_DEPTH)
                    )
                    # Ensure PV is present before accessing
                    if "pv" in info and info["pv"]:
                        oracle_cache[fen] = info["pv"][0].uci()
                    else:
                        print(f"[WARNING] Oracle found no principal variation for FEN: {fen}")
                except chess.engine.EngineTerminatedError as e:
                    print(f"[ERROR] Oracle engine terminated unexpectedly: {e}")
                    # Attempt to restart or exit gracefully
                    return oracle_cache # Return what we have so far
    except FileNotFoundError:
        print(f"[ERROR] Oracle engine not found at: {ORACLE_ENGINE_PATH}")
        return oracle_cache
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during oracle generation: {e}")
        return oracle_cache


    save_oracle_cache(oracle_cache)
    return oracle_cache
    
def run_engine_evaluations(oracle_cache):
    """
    Runs all benchmark engines against the oracle positions and logs results.
    """
    print("[INFO] Beginning engine evaluations...")

    # Load engine list
    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Engines CSV file not found at: {ENGINES_CSV_PATH}")
        return

    # Resume log if it exists
    if os.path.exists(GRANULAR_LOG_FILE):
        log_df = pd.read_csv(GRANULAR_LOG_FILE)
        completed = set(zip(log_df["engine_name"], log_df["fen"]))
        print(f"[RESUME] Loaded {len(log_df)} previously saved evaluations.")
    else:
        log_df = pd.DataFrame(columns=["engine_name", "fen", "score"])
        completed = set()

    # Iterate over each engine defined in the CSV file
    for _, engine_row in engines_df.iterrows():
        engine_name = engine_row["engine_name"]
        engine_path = engine_row["path"]
        uci_options_str = engine_row.get("uci_options", "{}")

        if engine_name == "stockfish_full":  # Skip the oracle engine itself
            continue

        print(f"\n[ENGINE] Analyzing with {engine_name}...")

        try:
            # *** FIX 1: Handle multiple limit types (Nodes, Depth, etc.) ***
            uci_options = json.loads(uci_options_str)
            limit_args = {}
            if "Nodes" in uci_options:
                limit_args["nodes"] = uci_options["Nodes"]
            if "Depth" in uci_options:
                limit_args["depth"] = uci_options["Depth"]

            # If no recognized limits are defined in the options, skip this engine.
            if not limit_args:
                print(f"[INFO] Skipping {engine_name} as it has no 'Nodes' or 'Depth' limit defined in its uci_options.")
                continue
            
            # Create the limit object to pass to the engine
            limit = chess.engine.Limit(**limit_args)

            with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
                # Use dual-core for strong engines, but single-core for Maia
                if "maia" in engine_name.lower():
                    engine.configure({"Threads": 1})
                else:
                    engine.configure({"Threads": 2})

                new_results = []
                for fen, oracle_best_value in tqdm(oracle_cache.items(), desc=f"Analyzing {engine_name}"):
                    if (engine_name, fen) in completed:
                        continue  # Skip positions already analyzed

                    board = chess.Board(fen)
                    
                    # *** FIX 2: Handle old cache format where the whole PV list might be stored ***
                    if isinstance(oracle_best_value, list):
                        # If the cached value is a list, assume it's a PV and take the first move.
                        if not oracle_best_value:
                            print(f"[WARNING] Empty PV list for FEN {fen} in cache. Skipping.")
                            continue
                        oracle_best_uci = oracle_best_value[0]
                    else:
                        # Otherwise, assume it's already a single UCI move string.
                        oracle_best_uci = oracle_best_value
                    
                    try:
                        oracle_best_move = chess.Move.from_uci(oracle_best_uci)
                    except ValueError:
                        print(f"[ERROR] Invalid UCI string '{oracle_best_uci}' for FEN {fen} in cache. Skipping.")
                        continue

                    try:
                        # Use the dynamically created limit object for analysis
                        info = engine.analyse(board, limit)
                        best_move = info.get("pv")[0] if info.get("pv") else None
                    except Exception as e:
                        print(f"[ERROR] {engine_name} failed on FEN {fen}: {e}")
                        continue

                    # Score is 1 if the engine's best move matches the oracle's
                    score = 1 if best_move == oracle_best_move else 0
                    new_results.append({"engine_name": engine_name, "fen": fen, "score": score})

                # Append all new results for this engine in one go for efficiency
                if new_results:
                    log_df = pd.concat([log_df, pd.DataFrame(new_results)], ignore_index=True)
                    # Save after each engine is finished
                    log_df.to_csv(GRANULAR_LOG_FILE, index=False)
                    print(f"[SAVE] Saved {len(new_results)} new results for {engine_name}.")

        except FileNotFoundError:
            print(f"[ERROR] Engine executable not found for {engine_name} at: {engine_path}")
        except json.JSONDecodeError:
            print(f"[ERROR] Could not parse uci_options for {engine_name}: '{uci_options_str}'")
        except Exception as e:
            print(f"[ERROR] A critical error occurred while running {engine_name}: {e}")

    # Final save
    log_df.to_csv(GRANULAR_LOG_FILE, index=False)
    print(f"\n[DONE] All evaluations saved to {GRANULAR_LOG_FILE}")


# ==============================================================================
# --- Main ---
# ==============================================================================
def main():
    """Main function to run the script."""
    print("--- Starting Rating Estimation Script ---")

    # Load oracle cache
    oracle_cache = load_oracle_cache()
    print(f"[CACHE] Found {len(oracle_cache)} cached positions.")

    # If cache has fewer positions than the target, sample more from the PGN.
    if len(oracle_cache) < TARGET_POSITIONS:
        print(f"[CACHE] Cache has {len(oracle_cache)} positions, which is less than the target of {TARGET_POSITIONS}. Sampling more...")
        new_positions = sample_positions_from_pgn(PGN_FILE, per_game=POSITIONS_PER_GAME)
        
        # Add only unique new positions to the list for oracle analysis
        positions_to_analyze = [p for p in list(set(new_positions)) if p not in oracle_cache]
        
        print(f"[INFO] Sampled {len(positions_to_analyze)} new unique positions from PGN.")

        if positions_to_analyze:
            oracle_cache = generate_oracle_moves(positions_to_analyze, oracle_cache)
            print(f"[DONE] Oracle move generation complete. Cache now has {len(oracle_cache)} positions.")
        else:
            print("[INFO] No new unique positions were found to add to the oracle cache.")
    else:
        print(f"[INFO] Using cached oracle moves ({len(oracle_cache)} positions).")

    # Proceed with engine analysis using the positions in the oracle cache
    if oracle_cache:
        print(f"[INFO] Proceeding with engine analysis using {len(oracle_cache)} oracle positions...")
        run_engine_evaluations(oracle_cache)
    else:
        print("[ERROR] No positions in oracle cache. Cannot run evaluations. Exiting.")


if __name__ == "__main__":
    main()
