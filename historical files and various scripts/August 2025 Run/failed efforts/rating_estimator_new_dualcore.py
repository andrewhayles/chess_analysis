import chess
import chess.pgn
import chess.engine
import pandas as pd
import time
import sys
import os
import json
from scipy.stats import linregress
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- Player and Game Configuration ---
PLAYER_NAME_IN_PGN = "Desjardins373"
PLAYER_PGN_PATH = "chessgames_august2025.pgn"

# --- Optimal Method Configuration ---
CHOSEN_METHOD = {
    "name": "Top 3 Moves, Linear Weights",
    "num_moves": 3,
    "weights": [1.0, 0.5, 0.25]
}

# --- File Paths ---
ENGINES_CSV_PATH = "real_engines.csv"
GRANULAR_LOG_PATH = "granular_analysis_log.csv"
OUTPUT_GRAPH_PATH = "player_rating_estimate_final.png"
ORACLE_CACHE_PATH = "oracle_cache.json"
STATUS_HISTORY_LOG_PATH = "status_history.csv"

# --- Default Engine Settings ---
ORACLE_ENGINE_NAME = "stockfish_full_1"
ORACLE_ANALYSIS_DEPTH = 22
ORACLE_ANALYSIS_TIMEOUT = 600
ENGINE_THREADS = 2 # Can be higher for single-core analysis
DEFAULT_TEST_DEPTH = 9
DEFAULT_TEST_TIMEOUT = 0.05 

# --- Engine-Specific Configuration Overrides ---
ENGINE_CONFIG_OVERRIDES = {

}

# ==============================================================================
# --- Core Logic ---
# ==============================================================================

def open_engine(path):
    """Opens a chess engine, handling potential startup issues."""
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        stderr_pipe = subprocess.DEVNULL if "leela" in path.lower() else None
        return chess.engine.SimpleEngine.popen_uci(path, stderr=stderr_pipe, startupinfo=startupinfo)
    except Exception as e:
        print(f"Error opening engine at {path}: {e}", file=sys.stderr)
        return None

def get_positions_from_pgn(pgn_path, player_name):
    """
    Extracts a dictionary mapping all of a player's FENs to their moves from a PGN file.
    This is used to look up the player's move for a given position.
    """
    if not os.path.exists(pgn_path):
        print(f"Error: PGN file not found at '{pgn_path}'.", file=sys.stderr)
        return {}
        
    player_move_map = {}
    print(f"Scanning PGN: {os.path.basename(pgn_path)} for all of {player_name}'s moves...")
    with open(pgn_path, 'r', errors='ignore') as pgn:
        while True:
            try:
                game = chess.pgn.read_game(pgn)
                if game is None: break

                white_player = game.headers.get("White", "")
                black_player = game.headers.get("Black", "")
                if player_name not in white_player and player_name not in black_player:
                    continue

                board = game.board()
                is_player_white = player_name in white_player
                
                for move in game.mainline_moves():
                    is_player_turn = (is_player_white and board.turn == chess.WHITE) or \
                                     (not is_player_white and board.turn == chess.BLACK)
                    
                    if is_player_turn:
                        player_move_map[board.fen()] = move.uci()
                    
                    board.push(move)
            except Exception as e:
                print(f"Skipping a game due to a parsing error: {e}", file=sys.stderr)
                continue
    return player_move_map

def get_move_score(move_to_check, oracle_moves, weights):
    """Calculates the score for a move based on the oracle's ranking."""
    score = 0.0
    for i, oracle_move in enumerate(oracle_moves):
        if move_to_check == oracle_move:
            if i < len(weights):
                score = weights[i]
            break
    return score

def main():
    """Main function to run the position-by-position analysis."""
    print("--- Starting Single-Core Rating Estimation Script ---")

    # 1. Load Data
    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Engines CSV file not found at '{ENGINES_CSV_PATH}'. Exiting.", file=sys.stderr)
        return

    # 2. Load Oracle Cache - THIS NOW DEFINES THE SCOPE OF WORK
    oracle_cache = {}
    if os.path.exists(ORACLE_CACHE_PATH):
        with open(ORACLE_CACHE_PATH, 'r') as f:
            try:
                oracle_cache = json.load(f)
                print(f"Loaded {len(oracle_cache)} positions from oracle cache. These will be analyzed.")
            except json.JSONDecodeError:
                print("Oracle cache file is corrupted. Exiting.", file=sys.stderr)
                return
    else:
        print(f"Error: Oracle Cache not found at '{ORACLE_CACHE_PATH}'. Cannot proceed.", file=sys.stderr)
        return

    # FIX: The set of positions to process is now taken directly from the cache.
    all_fens_to_process = set(oracle_cache.keys())
    if not all_fens_to_process:
        print("Oracle cache is empty. No positions to analyze.")
        return

    # Load all player moves from PGN to find matches for the cached positions.
    player_move_map = get_positions_from_pgn(PLAYER_PGN_PATH, PLAYER_NAME_IN_PGN)
    
    # 3. Verify Oracle Cache is sufficient (This section is now informational)
    fens_needing_oracle = list(all_fens_to_process - set(oracle_cache.keys()))
    if fens_needing_oracle:
        # This block should not be reached with the new logic, but is kept as a safeguard.
        print(f"Error: Mismatch in logic. Found positions needing oracle analysis.", file=sys.stderr)
        return
    else:
        print("\nOracle cache is sufficient. No new positions will be cached.")

    # 4. Main Analysis Loop (Single-Core)
    print(f"\n-- Performing analyses on {len(all_fens_to_process)} positions --")
    
    benchmark_engine_info = engines_df[engines_df['engine_name'] != ORACLE_ENGINE_NAME]
    benchmark_engines = {}
    for _, row in benchmark_engine_info.iterrows():
        engine = open_engine(row['path'])
        if engine:
            engine.configure({"Threads": ENGINE_THREADS})
            benchmark_engines[row['engine_name']] = engine

    if not benchmark_engines:
        print("Failed to initialize any benchmark engines. Exiting.", file=sys.stderr)
        return

    if os.path.exists(GRANULAR_LOG_PATH):
        os.remove(GRANULAR_LOG_PATH)
        print(f"Removed old log file: '{GRANULAR_LOG_PATH}'")

    all_results = []
    
    with tqdm(total=len(all_fens_to_process) * (len(benchmark_engines) + 1), desc="Analyzing Positions") as pbar:
        for fen in all_fens_to_process:
            board = chess.Board(fen)
            oracle_moves = oracle_cache.get(fen, [])

            for name, engine in benchmark_engines.items():
                try:
                    config = ENGINE_CONFIG_OVERRIDES.get(name, {})
                    depth = config.get('depth', DEFAULT_TEST_DEPTH)
                    timeout = config.get('time', DEFAULT_TEST_TIMEOUT)
                    limit = chess.engine.Limit(depth=depth, time=timeout)
                    result = engine.play(board, limit)
                    score = get_move_score(result.move.uci(), oracle_moves, CHOSEN_METHOD['weights'])
                except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
                    print(f"\nWarning: Engine '{name}' failed on FEN {fen}. Error: {e}", file=sys.stderr)
                    score = 0.0
                all_results.append({'fen': fen, 'engine_name': name, 'score': score})
                pbar.update(1)

            player_move = player_move_map.get(fen)
            if player_move:
                player_score = get_move_score(player_move, oracle_moves, CHOSEN_METHOD['weights'])
                all_results.append({'fen': fen, 'engine_name': 'player', 'score': player_score})
            pbar.update(1)

    print("\nClosing benchmark engines...")
    for engine in benchmark_engines.values():
        engine.quit()

    final_log_df = pd.DataFrame(all_results)
    final_log_df.to_csv(GRANULAR_LOG_PATH, index=False)
    print(f"All analyses complete. Log saved to '{GRANULAR_LOG_PATH}'.")

    # 5. Final Report and Graph
    print("\n--- Generating Final Report ---")
    avg_scores_df = final_log_df.groupby('engine_name')['score'].mean().reset_index()
    avg_scores_df.rename(columns={'score': 'average_hit_score'}, inplace=True)

    player_avg_score_row = avg_scores_df[avg_scores_df['engine_name'] == 'player']
    if player_avg_score_row.empty:
        print("No data for player found in log. Cannot estimate rating."); return
    player_avg_score = player_avg_score_row.iloc[0]['average_hit_score']
    
    engine_avg_scores = avg_scores_df[avg_scores_df['engine_name'] != 'player']
    final_df = pd.merge(benchmark_engine_info, engine_avg_scores, on='engine_name')
    
    if len(final_df) < 2:
        print("Not enough benchmark engine data to create a rating estimate."); return

    if final_df['average_hit_score'].nunique() > 1:
        slope, intercept, r_value, _, _ = linregress(final_df['average_hit_score'], final_df['rating'])
        final_r_squared = r_value ** 2
        final_player_rating = (slope * player_avg_score) + intercept
        
        print(f"\n--- Final Results ---")
        print(f"Analysis complete over {final_log_df['fen'].nunique()} positions.")
        print(f"Player's Final Average Score: {player_avg_score:.4f}")
        print(f"Final Estimated Rating for {PLAYER_NAME_IN_PGN}: {final_player_rating:.0f}")
        print(f"Final R-squared: {final_r_squared:.4f}")

        plt.figure(figsize=(12, 8))
        sns.regplot(x='rating', y='average_hit_score', data=final_df, ci=None, line_kws={'color':'red', 'linestyle':'--'}, label='Engine Trend')
        for _, row in final_df.iterrows():
            plt.scatter(row['rating'], row['average_hit_score'], s=80)
            plt.text(row['rating'] + 10, row['average_hit_score'], row['engine_name'], fontsize=9)

        plt.scatter(final_player_rating, player_avg_score, color='gold', s=200, edgecolor='black', zorder=5, label=f'You ({PLAYER_NAME_IN_PGN})')
        plt.text(final_player_rating + 10, player_avg_score, 'You', fontsize=11, weight='bold')

        plt.title(f"Final Performance Analysis vs. Engine Rating\nMethod: {CHOSEN_METHOD['name']}", fontsize=16)
        plt.xlabel("Engine Rating (Elo)", fontsize=12)
        plt.ylabel("Average Hit Score", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.text(0.05, 0.95, f'$R^2 = {final_r_squared:.4f}$\nFinal Est. Rating: {final_player_rating:.0f}', 
                 transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
                 
        plt.savefig(OUTPUT_GRAPH_PATH)
        print(f"\nGraph saved to: {OUTPUT_GRAPH_PATH}")
    else:
        print("\nCould not generate final report: All benchmark engines have the same average score.")

    print("--- Script Finished ---")


if __name__ == "__main__":
    main()
