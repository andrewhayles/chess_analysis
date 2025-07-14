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
# IMPORTANT: Update this path to your PGN file
PLAYER_PGN_PATH = "goalgames.pgn" 

# --- Optimal Method Configuration ---
# Defines how scores are calculated based on matching the Oracle's top moves.
CHOSEN_METHOD = {
    "name": "Top 3 Moves, Linear Weights",
    "num_moves": 3,
    "weights": [1.0, 0.5, 0.25] # Score for 1st, 2nd, 3rd best move
}

# --- Analysis Control ---
START_MOVE = 10 # Start analysis from this move number in each game.
POSITIONS_PER_GAME = 5 # Set to None to analyze all player moves after START_MOVE.

# --- File Paths ---
# IMPORTANT: Update this path to your engines file
ENGINES_CSV_PATH = "real_engines.csv"
# *** NEW ***: This is the primary log file for resuming. It logs every single analysis.
GRANULAR_LOG_PATH = "granular_analysis_log.csv"
OUTPUT_GRAPH_PATH = "player_rating_estimate_final.png"
ORACLE_CACHE_PATH = "oracle_cache.json"
# A log file that is constantly overwritten with the latest status.
LIVE_STATUS_LOG_PATH = "live_status_log.txt"


# --- Engine Settings ---
ORACLE_ENGINE_NAME = "stockfish_full_1" # The strongest engine, used as the 'ground truth'.
ORACLE_ANALYSIS_DEPTH = 22
ORACLE_ANALYSIS_TIMEOUT = 600 # Seconds for Oracle analysis
ENGINE_THREADS = 2 # Threads per engine instance.
TEST_ANALYSIS_DEPTH = 12 # Depth for benchmark engines and player move analysis.
TEST_ANALYSIS_TIMEOUT = 30 # Seconds for benchmark engine analysis

# ==============================================================================
# --- Core Logic ---
# ==============================================================================

def open_engine(path):
    """Opens a chess engine, handling potential startup issues."""
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    stderr_pipe = subprocess.DEVNULL if "leela" in path.lower() else None
    try:
        return chess.engine.SimpleEngine.popen_uci(path, stderr=stderr_pipe, startupinfo=startupinfo)
    except Exception as e:
        print(f"Error opening engine at {path}: {e}", file=sys.stderr)
        return None

def get_positions_from_pgn(pgn_path, player_name, positions_per_game, start_move):
    """Extracts a list of FENs and the player's move from a PGN file."""
    if not os.path.exists(pgn_path):
        print(f"Error: PGN file not found at '{pgn_path}'.", file=sys.stderr)
        return []
        
    positions = []
    print(f"Processing PGN: {os.path.basename(pgn_path)} for player: {player_name}")
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
                positions_in_this_game = 0
                is_player_white = player_name in white_player

                for move in game.mainline_moves():
                    is_player_turn = (is_player_white and board.turn == chess.WHITE) or \
                                     (not is_player_white and board.turn == chess.BLACK)
                    
                    if is_player_turn and board.fullmove_number >= start_move:
                        if positions_per_game is not None and positions_in_this_game >= positions_per_game:
                            break
                        positions.append({"fen": board.fen(), "actual_move": move.uci()})
                        positions_in_this_game += 1
                    board.push(move)
            except Exception as e:
                print(f"Skipping a game due to a parsing error: {e}", file=sys.stderr)
                continue
    return positions

def get_move_score(move_to_check, oracle_moves, weights):
    """Calculates the score for a move based on the oracle's ranking."""
    score = 0.0
    for i, oracle_move in enumerate(oracle_moves):
        if move_to_check == oracle_move:
            score = weights[i]
            break
    return score

def write_live_status(log_path, pos_count, total_pos, r_squared, estimated_rating):
    """Overwrites a status file with the latest progress and results."""
    try:
        with open(log_path, 'w') as f:
            rating_str = f"{estimated_rating:.0f}" if estimated_rating is not None else "N/A (calculating...)"
            f.write(f"--- Live Analysis Status ---\n")
            f.write(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Progress: Analyzed scores from {pos_count} / {total_pos} positions.\n\n")
            f.write(f"Current R-squared: {r_squared:.4f}\n")
            f.write(f"Current Estimated Rating: {rating_str}\n")
    except Exception as e:
        print(f"Warning: Could not write to live status log: {e}", file=sys.stderr)


def main():
    """Main function to run the position-by-position analysis."""
    print("--- Starting Position-by-Position Rating Estimation Script ---")
    
    # --- 1. Load Engine and Game Data ---
    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Engines CSV file not found at '{ENGINES_CSV_PATH}'. Exiting.", file=sys.stderr)
        return

    all_player_positions = get_positions_from_pgn(PLAYER_PGN_PATH, PLAYER_NAME_IN_PGN, POSITIONS_PER_GAME, START_MOVE)
    if not all_player_positions:
        print(f"No positions found for player '{PLAYER_NAME_IN_PGN}'. Exiting."); return
    print(f"Found {len(all_player_positions)} total positions for player '{PLAYER_NAME_IN_PGN}'.")

    # --- 2. Load Caches and Check Progress ---
    oracle_row = engines_df[engines_df['engine_name'] == ORACLE_ENGINE_NAME]
    if oracle_row.empty:
        print(f"Error: Oracle engine '{ORACLE_ENGINE_NAME}' not found in {ENGINES_CSV_PATH}."); return
    
    benchmark_engine_info = engines_df[engines_df['engine_name'] != ORACLE_ENGINE_NAME]
    engine_names = list(benchmark_engine_info['engine_name'])
    
    oracle_cache = {}
    if os.path.exists(ORACLE_CACHE_PATH):
        with open(ORACLE_CACHE_PATH, 'r') as f:
            try:
                oracle_cache = json.load(f)
                print(f"Loaded {len(oracle_cache)} positions from Oracle cache.")
            except json.JSONDecodeError: print("Oracle cache file is corrupted, starting fresh.")

    # *** NEW ***: Resume logic using the granular log
    completed_items = set()
    if os.path.exists(GRANULAR_LOG_PATH):
        try:
            log_df = pd.read_csv(GRANULAR_LOG_PATH)
            if not log_df.empty:
                completed_items = set(zip(log_df['fen'], log_df['engine_name']))
                print(f"Found {len(completed_items)} previously completed analyses in '{GRANULAR_LOG_PATH}'.")
        except (pd.errors.EmptyDataError, KeyError):
             print(f"'{GRANULAR_LOG_PATH}' is empty or malformed. Starting fresh.")
             completed_items = set()
    
    # --- 3. Phase 1: Update Oracle Cache ---
    all_fens_to_process = {p['fen'] for p in all_player_positions}
    fens_needing_oracle = list(all_fens_to_process - set(oracle_cache.keys()))
    
    if fens_needing_oracle:
        print(f"\n--- Phase 1: Caching {len(fens_needing_oracle)} new positions with Oracle Engine ---")
        oracle_engine = open_engine(oracle_row.iloc[0]['path'])
        if not oracle_engine:
            print("Could not start Oracle engine. Exiting.", file=sys.stderr)
            return
        oracle_engine.configure({"Threads": ENGINE_THREADS})
        
        for fen in tqdm(fens_needing_oracle, desc="Oracle Caching"):
            board = chess.Board(fen)
            limit = chess.engine.Limit(depth=ORACLE_ANALYSIS_DEPTH, time=ORACLE_ANALYSIS_TIMEOUT)
            analysis = oracle_engine.analyse(board, limit, multipv=CHOSEN_METHOD['num_moves'])
            oracle_moves = [info['pv'][0].uci() for info in analysis]
            oracle_cache[fen] = oracle_moves
            with open(ORACLE_CACHE_PATH, 'w') as f: json.dump(oracle_cache, f)
        oracle_engine.quit()
        print("Oracle caching complete.")
    else:
        print("\nOracle cache is already up-to-date.")

    # --- 4. Main Analysis Loop ---
    items_to_analyze = []
    for pos in all_player_positions:
        # Check if player analysis for this FEN is done
        if (pos['fen'], 'player') not in completed_items:
            items_to_analyze.append({'type': 'player', 'fen': pos['fen'], 'actual_move': pos['actual_move']})
        # Check for each benchmark engine
        for name in engine_names:
            if (pos['fen'], name) not in completed_items:
                items_to_analyze.append({'type': 'engine', 'fen': pos['fen'], 'engine_name': name})

    if not items_to_analyze:
        print("All analyses are already complete. Generating final report.")
    else:
        print(f"\n--- Phase 2: Performing {len(items_to_analyze)} new analyses ---")
        benchmark_engines = {}
        for _, row in benchmark_engine_info.iterrows():
            engine = open_engine(row['path'])
            if engine:
                engine.configure({"Threads": ENGINE_THREADS})
                benchmark_engines[row['engine_name']] = engine
        
        if not benchmark_engines:
            print("Failed to initialize any benchmark engines. Exiting.", file=sys.stderr); return

        pbar = tqdm(total=len(items_to_analyze), desc="Benchmark Analysis", unit="analysis")
        header_written = len(completed_items) > 0

        for item in items_to_analyze:
            fen = item['fen']
            board = chess.Board(fen)
            oracle_moves = oracle_cache[fen]
            
            score = 0
            engine_name_for_log = ""

            if item['type'] == 'player':
                score = get_move_score(item['actual_move'], oracle_moves, CHOSEN_METHOD['weights'])
                engine_name_for_log = 'player'
            
            elif item['type'] == 'engine':
                engine_name = item['engine_name']
                engine_name_for_log = engine_name
                try:
                    limit = chess.engine.Limit(depth=TEST_ANALYSIS_DEPTH, time=TEST_ANALYSIS_TIMEOUT)
                    result = benchmark_engines[engine_name].play(board, limit)
                    score = get_move_score(result.move.uci(), oracle_moves, CHOSEN_METHOD['weights'])
                except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
                    print(f"\nWarning: Engine '{engine_name}' failed on FEN {fen}. Error: {e}", file=sys.stderr)
                    score = 0.0

            # *** NEW ***: Log every single completed item immediately
            log_entry = pd.DataFrame([{'fen': fen, 'engine_name': engine_name_for_log, 'score': score}])
            log_entry.to_csv(GRANULAR_LOG_PATH, mode='a', header=not header_written, index=False)
            header_written = True
            
            pbar.update(1)

        pbar.close()
        print("\nClosing benchmark engines...")
        for engine in benchmark_engines.values(): engine.quit()

    # --- 5. Final Report and Graph ---
    print("\n--- Generating Final Report from Granular Log ---")
    if not os.path.exists(GRANULAR_LOG_PATH):
        print("Log file not found. Cannot generate report."); return
        
    full_log_df = pd.read_csv(GRANULAR_LOG_PATH)
    if full_log_df.empty:
        print("Log file is empty. Cannot generate report."); return

    # Calculate average scores from the complete granular log
    avg_scores_df = full_log_df.groupby('engine_name')['score'].mean().reset_index()
    avg_scores_df.rename(columns={'score': 'average_hit_score'}, inplace=True)

    player_avg_score_row = avg_scores_df[avg_scores_df['engine_name'] == 'player']
    if player_avg_score_row.empty:
        print("No data for player found in log. Cannot estimate rating."); return
    player_avg_score = player_avg_score_row.iloc[0]['average_hit_score']
    
    # Merge with engine ratings
    engine_avg_scores = avg_scores_df[avg_scores_df['engine_name'] != 'player']
    final_df = pd.merge(benchmark_engine_info, engine_avg_scores, on='engine_name')
    
    if len(final_df) < 2:
        print("Not enough benchmark engine data to create a rating estimate."); return

    if final_df['average_hit_score'].nunique() > 1:
        slope, intercept, r_value, _, _ = linregress(final_df['average_hit_score'], final_df['rating'])
        final_r_squared = r_value ** 2
        final_player_rating = (slope * player_avg_score) + intercept
        
        print(f"\n--- Final Results ---")
        print(f"Analysis complete over {full_log_df['fen'].nunique()} positions.")
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
