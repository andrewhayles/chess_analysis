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

# --- Default Engine Settings ---
ORACLE_ENGINE_NAME = "stockfish_full_1"
ORACLE_ANALYSIS_DEPTH = 22
ORACLE_ANALYSIS_TIMEOUT = 600
ENGINE_THREADS = 2
DEFAULT_TEST_DEPTH = 5
DEFAULT_TEST_TIMEOUT = 0.05 

# --- Engine-Specific Configuration Overrides ---
# NEW: Added overrides for the rate-limited Stockfish engines to match Maia.
ENGINE_CONFIG_OVERRIDES = {
    "stockfish_elo_1350": {"time": 1.0},
    "stockfish_elo_1450": {"time": 1.0},
    "stockfish_elo_1550": {"time": 1.0},
    "stockfish_elo_1650": {"time": 1.0},
    "stockfish_elo_1750": {"time": 1.0},
    "stockfish_elo_1850": {"time": 1.0},
    "stockfish_elo_1950": {"time": 1.0},
    "stockfish_elo_2100": {"time": 1.0},
    "stockfish_elo_2300": {"time": 1.0},
    "stockfish_elo_2500": {"time": 1.0},
    "stockfish_elo_2700": {"time": 1.0},
}

# ==============================================================================
# --- Core Logic ---
# ==============================================================================

def open_engine(path, uci_options_str):
    """Opens a chess engine and applies any specified UCI options."""
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        stderr_pipe = subprocess.DEVNULL if "leela" in path.lower() else None
        engine = chess.engine.SimpleEngine.popen_uci(path, stderr=stderr_pipe, startupinfo=startupinfo)
        
        # NEW: Apply UCI options from the CSV
        if uci_options_str and uci_options_str != "{}":
            try:
                options = json.loads(uci_options_str)
                for name, value in options.items():
                    engine.configure({name: value})
                # print(f"Successfully configured engine {os.path.basename(path)} with: {options}")
            except json.JSONDecodeError:
                print(f"Warning: Could not parse UCI options for {path}: {uci_options_str}", file=sys.stderr)
        return engine
    except Exception as e:
        print(f"Error opening engine at {path}: {e}", file=sys.stderr)
        return None

def get_positions_from_pgn(pgn_path, player_name):
    """
    Extracts a dictionary mapping all of a player's FENs to their moves from a PGN file.
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
                if player_name not in game.headers.get("White", "") and player_name not in game.headers.get("Black", ""):
                    continue
                board = game.board()
                is_player_white = player_name in game.headers.get("White", "")
                for move in game.mainline_moves():
                    is_player_turn = (is_player_white and board.turn == chess.WHITE) or \
                                     (not is_player_white and board.turn == chess.BLACK)
                    if is_player_turn:
                        player_move_map[board.fen()] = move.uci()
                    board.push(move)
            except Exception:
                continue
    return player_move_map

def get_move_score(move_to_check, oracle_moves, weights):
    """Calculates the score for a move based on the oracle's ranking."""
    score = 0.0
    for i, oracle_move in enumerate(oracle_moves):
        if move_to_check == oracle_move and i < len(weights):
            score = weights[i]
            break
    return score

def main():
    """Main function to run the position-by-position analysis."""
    print("--- Starting Single-Core Rating Estimation Script ---")

    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Engines CSV file not found at '{ENGINES_CSV_PATH}'. Exiting.", file=sys.stderr)
        return

    try:
        with open(ORACLE_CACHE_PATH, 'r') as f:
            oracle_cache = json.load(f)
        print(f"Loaded {len(oracle_cache)} positions from oracle cache. These will be analyzed.")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: Oracle Cache not found or corrupted at '{ORACLE_CACHE_PATH}'.", file=sys.stderr)
        return

    all_fens_to_process = set(oracle_cache.keys())
    if not all_fens_to_process:
        print("Oracle cache is empty. No positions to analyze.")
        return

    player_move_map = get_positions_from_pgn(PLAYER_PGN_PATH, PLAYER_NAME_IN_PGN)
    
    print(f"\n-- Performing analyses on {len(all_fens_to_process)} positions --")
    
    benchmark_engine_info = engines_df[engines_df['engine_name'] != ORACLE_ENGINE_NAME]
    benchmark_engines = {}
    for _, row in benchmark_engine_info.iterrows():
        # Pass the UCI options string to the open_engine function
        engine = open_engine(row['path'], row.get('uci_options', '{}'))
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

    # Final report and graph generation (remains the same)
    print("\n--- Generating Final Report ---")
    avg_scores_df = final_log_df.groupby('engine_name')['score'].mean().reset_index()
    avg_scores_df.rename(columns={'score': 'average_hit_score'}, inplace=True)

    player_avg_score_row = avg_scores_df[avg_scores_df['engine_name'] == 'player']
    if player_avg_score_row.empty:
        print("No data for player found in log. Cannot estimate rating."); return
    player_avg_score = player_avg_score_row.iloc[0]['average_hit_score']
    
    engine_avg_scores = avg_scores_df[avg_scores_df['engine_name'] != 'player']
    final_df = pd.merge(benchmark_engine_info, engine_avg_scores, on='engine_name')
    
    if len(final_df) < 2 or final_df['average_hit_score'].nunique() <= 1:
        print("Not enough benchmark data or variance to create a rating estimate."); return

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

    print("--- Script Finished ---")


if __name__ == "__main__":
    main()
