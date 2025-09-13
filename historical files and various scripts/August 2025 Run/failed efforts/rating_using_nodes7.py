import chess
import chess.pgn
import chess.engine
import os
import json
import random
from tqdm import tqdm

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
    if os.path.exists(ORACLE_CACHE_FILE):
        with open(ORACLE_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_oracle_cache(cache):
    with open(ORACLE_CACHE_FILE, "w") as f:
        json.dump(cache, f)

def sample_positions_from_pgn(pgn_file, per_game=5):
    """Randomly sample legal positions from each game after move 10."""
    positions = []
    with open(pgn_file, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if game.headers.get("White") != PLAYER_NAME and game.headers.get("Black") != PLAYER_NAME:
                continue

            moves = list(game.mainline_moves())
            if len(moves) <= 10:
                continue

            # pick ply indices (not moves) after 10th move
            sample_indices = random.sample(
                range(10, len(moves)), 
                min(per_game, len(moves) - 10)
            )

            for idx in sample_indices:
                board = game.board()
                for move in moves[:idx+1]:
                    board.push(move)
                positions.append(board.fen())
    return positions


def generate_oracle_moves(positions, oracle_cache):
    """Generate oracle moves using Stockfish full-strength with depth/time limits."""
    print(f"[ORACLE] Generating oracle moves with Stockfish "
          f"(max depth {ORACLE_MAX_DEPTH}, max {ORACLE_TIME_LIMIT}s each, 2 threads).")

    with chess.engine.SimpleEngine.popen_uci(ORACLE_ENGINE_PATH) as oracle:
        oracle.configure({"Threads": 2})
        for fen in tqdm(positions, desc="Generating oracle moves"):
            if fen in oracle_cache:
                continue
            board = chess.Board(fen)
            # Limit both time and depth
            info = oracle.analyse(
                board,
                chess.engine.Limit(time=ORACLE_TIME_LIMIT, depth=ORACLE_MAX_DEPTH)
            )
            oracle_cache[fen] = info["pv"][0].uci()

    save_oracle_cache(oracle_cache)
    return oracle_cache
    
def run_engine_evaluations(oracle_cache):
    """
    Runs all benchmark engines against the oracle positions and logs results.
    """
    import pandas as pd

    print("[INFO] Beginning engine evaluations...")

    # Load engine list
    engines_df = pd.read_csv(ENGINES_CSV_PATH)

    # Resume log if it exists
    if os.path.exists(GRANULAR_LOG_FILE):
        log_df = pd.read_csv(GRANULAR_LOG_FILE)
        completed = set(zip(log_df["engine_name"], log_df["fen"]))
        print(f"[RESUME] Loaded {len(log_df)} previously saved evaluations.")
    else:
        log_df = pd.DataFrame(columns=["engine_name", "fen", "score"])
        completed = set()

    # Iterate over engines
    for _, engine_row in engines_df.iterrows():
        engine_name = engine_row["engine_name"]
        engine_path = engine_row["path"]

        if engine_name == "stockfish_full":  # skip oracle engine
            continue

        print(f"\n[ENGINE] Analyzing with {engine_name}...")

        try:
            with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
                # Use dual-core for strong engines, but single-core for Maia
                if "maia" in engine_name.lower():
                    engine.configure({"Threads": 1})
                else:
                    engine.configure({"Threads": 2})

                for fen, oracle_best in tqdm(oracle_cache.items(), desc=f"Analyzing {engine_name}"):
                    if (engine_name, fen) in completed:
                        continue  # already done

                    board = chess.Board(fen)
                    try:
                        info = engine.analyse(board, chess.engine.Limit(nodes=ENGINE_NODE_LIMIT))
                        best_move = info["pv"][0] if "pv" in info else None
                    except Exception as e:
                        print(f"[ERROR] {engine_name} failed on {fen}: {e}")
                        continue

                    # Score = 1 if engine matched oracle, 0 otherwise
                    score = 1 if best_move == oracle_best else 0

                    # Append new result
                    new_row = {"engine_name": engine_name, "fen": fen, "score": score}
                    log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)

                    # Periodically save to CSV
                    if len(log_df) % 100 == 0:
                        log_df.to_csv(GRANULAR_LOG_FILE, index=False)

        except Exception as e:
            print(f"[ERROR] Failed to run {engine_name}: {e}")

    # Final save
    log_df.to_csv(GRANULAR_LOG_FILE, index=False)
    print(f"[DONE] All evaluations saved to {GRANULAR_LOG_FILE}")


# ==============================================================================
# --- Main ---
# ==============================================================================
def main():
    print("--- Starting Rating Estimation Script ---")

    # Load oracle cache
    oracle_cache = load_oracle_cache()
    print(f"[CACHE] Found {len(oracle_cache)} cached positions.")

    # If not enough positions, expand cache
    if len(oracle_cache) < TARGET_POSITIONS:
        print(f"[CACHE] Only {len(oracle_cache)} cached positions. Sampling more...")
        new_positions = sample_positions_from_pgn(PGN_FILE, per_game=POSITIONS_PER_GAME)
        unique_positions = list(set(new_positions))  # deduplicate
        print(f"[INFO] Sampled {len(unique_positions)} new positions from PGN.")

        needed_positions = TARGET_POSITIONS - len(oracle_cache)
        selected_positions = unique_positions[:needed_positions]

        oracle_cache = generate_oracle_moves(selected_positions, oracle_cache)

        print(f"[DONE] Oracle move generation complete. Cache now has {len(oracle_cache)} positions.")
    else:
        print(f"[INFO] Using cached oracle moves ({len(oracle_cache)} positions).")

    # ✅ At this point we always continue
    print(f"[INFO] Proceeding with engine analysis using {len(oracle_cache)} oracle positions...")
    run_engine_evaluations(oracle_cache)


if __name__ == "__main__":
    main()
