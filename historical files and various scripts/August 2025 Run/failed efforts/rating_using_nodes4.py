import os
import sys
import chess
import chess.pgn
import chess.engine
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# --- Configuration ---
# ==============================================================================
PGN_FILE = "chessgames_august2025.pgn"
ENGINES_CSV_PATH = "real_engines.csv"
GRANULAR_LOG_PATH = "granular_analysis_log.csv"

PLAYER_NAME_IN_PGN = "Desjardins373"
ORACLE_ENGINE_PATH = "C:/Users/desja/Documents/my_programming/chess_analysis/engines/stockfish/stockfish-windows-x86-64-sse41-popcnt.exe"

# Oracle depth for “ground truth”
ORACLE_DEPTH = 22

# Toggle test mode: if True, only run a handful of positions to verify engine commands
TEST_MODE = True
TEST_POSITIONS = 5

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================
def load_positions_from_pgn(pgn_path, player_name):
    """Extract positions where the given player has to move."""
    positions = []
    with open(pgn_path, encoding="utf-8") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                if board.turn == (game.headers.get("White") == player_name):
                    positions.append(board.fen())
                board.push(move)
    return positions

def get_oracle_moves(engine_path, positions, depth):
    """Use a strong engine to generate the 'oracle' best move for each position."""
    oracle = chess.engine.SimpleEngine.popen_uci(engine_path)
    oracle_moves = {}
    for fen in tqdm(positions, desc="Generating oracle moves"):
        board = chess.Board(fen)
        result = oracle.play(board, chess.engine.Limit(depth=depth))
        oracle_moves[fen] = result.move
    oracle.quit()
    return oracle_moves

def run_engine_on_position(engine_name, engine_path, uci_options, board):
    """Run an engine with the given options and log the exact limits."""
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    # Apply UCI options
    limit_kwargs = {}
    for key, val in uci_options.items():
        if key.lower() == "depth":
            limit_kwargs["depth"] = int(val)
        elif key.lower() == "nodes":
            limit_kwargs["nodes"] = int(val)
        else:
            try:
                engine.configure({key: val})
            except Exception as e:
                print(f"[WARN] Engine {engine_name} rejected option {key}={val}: {e}")

    # Debug log
    print(f"[DEBUG] Engine {engine_name} running with limits: {limit_kwargs}", flush=True)

    # Run engine
    result = engine.play(board, chess.engine.Limit(**limit_kwargs))
    engine.quit()
    return result.move

# ==============================================================================
# --- Main Analysis ---
# ==============================================================================
def main():
    print("--- Starting Rating Estimation Script ---")

    # 1. Check for old log
    if os.path.exists(GRANULAR_LOG_PATH):
        print(f"[WARNING] Found existing log file at {GRANULAR_LOG_PATH}.")
        print(" If your previous runs had engines with wrong depth/node settings, you should DELETE this file before continuing.")
        print(" Otherwise, the script will resume using the old data.\n")

    # 2. Load player positions
    print(f"Scanning PGN: {PGN_FILE} for all of {PLAYER_NAME_IN_PGN}'s moves...")
    player_positions = load_positions_from_pgn(PGN_FILE, PLAYER_NAME_IN_PGN)
    print(f"Found {len(player_positions)} positions for {PLAYER_NAME_IN_PGN}.")

    # 3. Load engines
    engines_df = pd.read_csv(ENGINES_CSV_PATH)
    engines = []
    for _, row in engines_df.iterrows():
        name = row["engine_name"]
        path = row["path"]
        try:
            uci_opts = eval(row["uci_options"]) if pd.notna(row["uci_options"]) else {}
        except Exception:
            uci_opts = {}
        engines.append((name, path, row["rating"], uci_opts))

    # In test mode, skip Maia engines for faster debugging
    if TEST_MODE:
        engines = [e for e in engines if not e[0].startswith("maia_")]
        print("[TEST MODE] Maia engines skipped to avoid slow ONNX loading.")

    # 4. Resume log if exists
    if os.path.exists(GRANULAR_LOG_PATH):
        log_df = pd.read_csv(GRANULAR_LOG_PATH)
        done_positions = set(log_df["fen"].unique())
        print(f"Resuming from {len(log_df)} previously saved results.")
    else:
        log_df = pd.DataFrame(columns=["fen", "engine_name", "score"])
        done_positions = set()

    # 5. Select positions
    all_positions = [fen for fen in player_positions if fen not in done_positions]
    if TEST_MODE:
        all_positions = all_positions[:TEST_POSITIONS]
        print(f"[TEST MODE] Restricting to {len(all_positions)} positions for quick verification.")

    if not all_positions:
        print("All positions already processed. Exiting.")
        return

    oracle_moves = get_oracle_moves(ORACLE_ENGINE_PATH, all_positions, ORACLE_DEPTH)

    # 6. Analyze
    new_results = []
    for fen in tqdm(all_positions, desc="Analyzing Positions"):
        board = chess.Board(fen)
        oracle_move = oracle_moves[fen]

        # Player move
        player_move = board.peek() if board.move_stack else None
        if player_move is not None:
            score = 1 if player_move == oracle_move else 0
            new_results.append({"fen": fen, "engine_name": "player", "score": score})

        # Engines
        for name, path, rating, uci_opts in engines:
            try:
                move = run_engine_on_position(name, path, uci_opts, board)
                score = 1 if move == oracle_move else 0
                new_results.append({"fen": fen, "engine_name": name, "score": score})
            except Exception as e:
                print(f"[ERROR] Engine {name} failed on FEN {fen}: {e}")

    # Save results
    if new_results:
        pd.DataFrame(new_results).to_csv(
            GRANULAR_LOG_PATH, mode="a", header=not os.path.exists(GRANULAR_LOG_PATH), index=False
        )

    print(f"--- Analysis complete. Results saved to {GRANULAR_LOG_PATH} ---")
    if TEST_MODE:
        print("✅ Test mode complete. Please check the [DEBUG] logs above to confirm Stockfish depth/nodes limits.")

if __name__ == "__main__":
    main()
