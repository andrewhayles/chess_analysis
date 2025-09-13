import chess
import chess.pgn
import chess.engine
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import time

# ==============================================================================
# --- Configuration ---
# ==============================================================================
PGN_FILE = "chessgames_august2025.pgn"
GRANULAR_LOG_PATH = "granular_analysis_log.csv"
ENGINES_CSV_PATH = "real_engines.csv"

BATCH_SIZE = 100     # how many results before writing to CSV
PLAYER_NAME = "Desjardins373"
ORACLE_ENGINE_PATH = r"C:\Users\desja\Documents\my_programming\chess_analysis\engines\stockfish\stockfish-windows-x86-64-sse41-popcnt.exe"
ORACLE_DEPTH = 22    # oracle search depth

TEST_MODE = False
TEST_LIMIT = 5       # how many positions max in test mode

# ==============================================================================
# --- Utility Functions ---
# ==============================================================================
def get_oracle_moves(positions):
    """Use Stockfish (deep depth) to generate the oracle move for each FEN."""
    oracle = chess.engine.SimpleEngine.popen_uci(ORACLE_ENGINE_PATH)
    oracle_moves = {}
    for fen in tqdm(positions, desc="Generating oracle moves"):
        board = chess.Board(fen)
        info = oracle.analyse(board, chess.engine.Limit(depth=ORACLE_DEPTH))
        oracle_moves[fen] = info["pv"][0]  # principal variation move
    oracle.quit()
    return oracle_moves

def run_engine(engine_path, limits, fen):
    """Run a given engine with specified limits on a single FEN."""
    board = chess.Board(fen)
    try:
        with chess.engine.SimpleEngine.popen_uci(engine_path) as eng:
            info = eng.analyse(board, limits)
            return info["pv"][0] if "pv" in info else None
    except Exception as e:
        print(f"[ERROR] Engine {engine_path} failed on FEN {fen}: {e}")
        return None

# ==============================================================================
# --- Main Analysis Function ---
# ==============================================================================
def main():
    print("--- Starting Rating Estimation Script ---")

    # --------------------------------------------------------------------------
    # Load PGN and collect all player moves
    # --------------------------------------------------------------------------
    if not os.path.exists(PGN_FILE):
        print(f"Error: PGN file '{PGN_FILE}' not found.")
        return

    pgn = open(PGN_FILE, encoding="utf-8")
    positions = []
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        if game.headers.get("White") == PLAYER_NAME or game.headers.get("Black") == PLAYER_NAME:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                positions.append(board.fen())

    print(f"Found {len(positions)} positions for {PLAYER_NAME}.")

    if TEST_MODE:
        positions = positions[:TEST_LIMIT]
        print(f"[TEST MODE] Restricting to {len(positions)} positions for quick verification.")

    # --------------------------------------------------------------------------
    # Prepare engines list
    # --------------------------------------------------------------------------
    engines_df = pd.read_csv(ENGINES_CSV_PATH)
    engines = engines_df.to_dict("records")

    # --------------------------------------------------------------------------
    # Oracle moves
    # --------------------------------------------------------------------------
    unique_positions = list(set(positions))
    print(f"-- Performing analyses on {len(unique_positions)} unique positions --")
    oracle_moves = get_oracle_moves(unique_positions)

    # --------------------------------------------------------------------------
    # Resume if log already exists
    # --------------------------------------------------------------------------
    if os.path.exists(GRANULAR_LOG_PATH):
        old_df = pd.read_csv(GRANULAR_LOG_PATH)
        completed = set(zip(old_df["fen"], old_df["engine_name"]))
        print(f"Resuming from {len(old_df)} previously saved results.")
    else:
        old_df = pd.DataFrame(columns=["fen", "engine_name", "score"])
        completed = set()

    # --------------------------------------------------------------------------
    # Main analysis loop
    # --------------------------------------------------------------------------
    results = []
    for fen in tqdm(unique_positions, desc="Analyzing Positions"):
        for engine in engines:
            eng_name = engine["engine_name"]
            if (fen, eng_name) in completed:
                continue

            limits = {}
            if pd.notna(engine.get("depth")):
                limits = chess.engine.Limit(depth=int(engine["depth"]))
            elif pd.notna(engine.get("nodes")):
                limits = chess.engine.Limit(nodes=int(engine["nodes"]))

            # --- Run engine ---
            move = run_engine(engine["engine_path"], limits, fen)

            # --- Score (1 = matched oracle, 0 = not) ---
            score = 1 if move == oracle_moves[fen] else 0

            results.append({"fen": fen, "engine_name": eng_name, "score": score})

            # --- Batch write ---
            if len(results) >= BATCH_SIZE:
                new_df = pd.DataFrame(results)
                new_df.to_csv(GRANULAR_LOG_PATH, mode="a", header=not os.path.exists(GRANULAR_LOG_PATH), index=False)
                results = []

                # --- Per-engine accuracy tracker ---
                try:
                    log_df = pd.read_csv(GRANULAR_LOG_PATH)
                    engine_stats = (
                        log_df.groupby("engine_name")["score"]
                        .agg(["mean", "count", "sum"])
                        .reset_index()
                    )
                    print("\n[STATS] Current engine accuracies:")
                    for _, row in engine_stats.iterrows():
                        accuracy = 100 * row["mean"]
                        hits = int(row["sum"])
                        total = int(row["count"])
                        print(f"   {row['engine_name']:20s} → {accuracy:5.1f}% ({hits}/{total})")
                    print("-" * 60)
                except Exception as e:
                    print(f"[WARN] Could not compute engine stats: {e}")

    # --------------------------------------------------------------------------
    # Final flush
    # --------------------------------------------------------------------------
    if results:
        new_df = pd.DataFrame(results)
        new_df.to_csv(GRANULAR_LOG_PATH, mode="a", header=not os.path.exists(GRANULAR_LOG_PATH), index=False)

    print("Analysis complete. Results saved to", GRANULAR_LOG_PATH)


if __name__ == "__main__":
    main()
