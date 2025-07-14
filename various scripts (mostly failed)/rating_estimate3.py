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
# This new log file will store the detailed position-by-position analysis.
OUTPUT_LOG_PATH = "position_analysis_log.csv"
OUTPUT_GRAPH_PATH = "player_rating_estimate_final.png"
# *** NEW ***: Cache for Oracle results to avoid re-analyzing positions.
ORACLE_CACHE_PATH = "oracle_cache.json"

# --- Engine Settings ---
ORACLE_ENGINE_NAME = "stockfish_full_1" # The strongest engine, used as the 'ground truth'.
ORACLE_ANALYSIS_DEPTH = 22
ORACLE_ANALYSIS_TIMEOUT = 60 # Seconds for Oracle analysis
ENGINE_THREADS = 2 # Threads per engine instance.
TEST_ANALYSIS_DEPTH = 12 # Depth for benchmark engines and player move analysis.
# *** NEW ***: Timeout for benchmark engine analysis.
TEST_ANALYSIS_TIMEOUT = 30 # Seconds for benchmark engine analysis

# ==============================================================================
# --- Core Logic ---
# ==============================================================================

def open_engine(path):
    """Opens a chess engine, handling potential startup issues."""
    startupinfo = None
    # This helps hide console windows on Windows OS.
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    # Suppress stderr for noisy engines like Leela.
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
                # This can happen with malformed PGNs.
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

def main():
    """Main function to run the position-by-position analysis."""
    print("--- Starting Position-by-Position Rating Estimation Script ---")
    
    # --- 1. Load Engine and Game Data ---
    try:
        engines_df = pd.read_csv(ENGINES_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Engines CSV file not found at '{ENGINES_CSV_PATH}'. Exiting.", file=sys.stderr)
        return

    player_positions = get_positions_from_pgn(PLAYER_PGN_PATH, PLAYER_NAME_IN_PGN, POSITIONS_PER_GAME, START_MOVE)
    if not player_positions:
        print(f"No positions found for player '{PLAYER_NAME_IN_PGN}'. Exiting."); return
    print(f"Found {len(player_positions)} positions to analyze for player '{PLAYER_NAME_IN_PGN}'.")

    # --- 2. Initialize Engines and Oracle Cache ---
    print("Initializing engines...")
    oracle_row = engines_df[engines_df['engine_name'] == ORACLE_ENGINE_NAME]
    if oracle_row.empty:
        print(f"Error: Oracle engine '{ORACLE_ENGINE_NAME}' not found in {ENGINES_CSV_PATH}."); return
    
    oracle_engine = open_engine(oracle_row.iloc[0]['path'])
    if oracle_engine:
        oracle_engine.configure({"Threads": ENGINE_THREADS})

    benchmark_engines = {}
    benchmark_engine_info = engines_df[engines_df['engine_name'] != ORACLE_ENGINE_NAME]
    for _, row in benchmark_engine_info.iterrows():
        engine = open_engine(row['path'])
        if engine:
            engine.configure({"Threads": ENGINE_THREADS})
            benchmark_engines[row['engine_name']] = engine
    
    if not oracle_engine or not benchmark_engines:
        print("Failed to initialize one or more engines. Exiting.", file=sys.stderr)
        return
        
    # Load Oracle Cache
    oracle_cache = {}
    if os.path.exists(ORACLE_CACHE_PATH):
        with open(ORACLE_CACHE_PATH, 'r') as f:
            try:
                oracle_cache = json.load(f)
                print(f"Loaded {len(oracle_cache)} positions from Oracle cache.")
            except json.JSONDecodeError:
                print("Oracle cache file is corrupted, starting fresh.")


    # --- 3. Main Analysis Loop ---
    all_results = []
    running_totals = {name: {'score_sum': 0, 'count': 0} for name in benchmark_engines.keys()}
    running_totals['player'] = {'score_sum': 0, 'count': 0}

    header_written = os.path.exists(OUTPUT_LOG_PATH) and os.path.getsize(OUTPUT_LOG_PATH) > 0
    
    print(f"\nAnalyzing {len(player_positions)} positions...")
    for pos_data in tqdm(player_positions, desc="Analyzing Positions"):
        fen = pos_data['fen']
        player_actual_move = pos_data['actual_move']
        board = chess.Board(fen)
        
        current_pos_results = {'fen': fen, 'player_move': player_actual_move}

        # a) Oracle Analysis (with caching)
        if fen in oracle_cache:
            oracle_moves = oracle_cache[fen]
        else:
            limit = chess.engine.Limit(depth=ORACLE_ANALYSIS_DEPTH, time=ORACLE_ANALYSIS_TIMEOUT)
            analysis = oracle_engine.analyse(board, limit, multipv=CHOSEN_METHOD['num_moves'])
            oracle_moves = [info['pv'][0].uci() for info in analysis]
            oracle_cache[fen] = oracle_moves
            # Save cache after each new analysis
            with open(ORACLE_CACHE_PATH, 'w') as f:
                json.dump(oracle_cache, f)

        # b) Score Player's Move
        player_score = get_move_score(player_actual_move, oracle_moves, CHOSEN_METHOD['weights'])
        current_pos_results['player_score'] = player_score
        running_totals['player']['score_sum'] += player_score
        running_totals['player']['count'] += 1

        # c) Score Benchmark Engines' Moves
        limit = chess.engine.Limit(depth=TEST_ANALYSIS_DEPTH, time=TEST_ANALYSIS_TIMEOUT)
        for name, engine in benchmark_engines.items():
            try:
                result = engine.play(board, limit)
                engine_score = get_move_score(result.move.uci(), oracle_moves, CHOSEN_METHOD['weights'])
                current_pos_results[f'{name}_score'] = engine_score
                running_totals[name]['score_sum'] += engine_score
                running_totals[name]['count'] += 1
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
                print(f"Warning: Engine '{name}' failed on FEN {fen}. Skipping. Error: {e}", file=sys.stderr)
                current_pos_results[f'{name}_score'] = 0.0 # Assign a zero score on failure
            
        # d) Calculate Running Average Rating
        avg_scores_data = []
        for name, totals in running_totals.items():
            if totals['count'] > 0 and name != 'player':
                avg_score = totals['score_sum'] / totals['count']
                rating = benchmark_engine_info[benchmark_engine_info['engine_name'] == name].iloc[0]['rating']
                avg_scores_data.append({'engine_name': name, 'average_hit_score': avg_score, 'rating': rating})
        
        estimated_rating = None
        if len(avg_scores_data) >= 2:
            regression_df = pd.DataFrame(avg_scores_data)
            slope, intercept, _, _, _ = linregress(regression_df['average_hit_score'], regression_df['rating'])
            player_avg_score = running_totals['player']['score_sum'] / running_totals['player']['count']
            estimated_rating = (slope * player_avg_score) + intercept
        
        current_pos_results['player_running_avg_rating'] = f"{estimated_rating:.0f}" if estimated_rating is not None else "N/A"
        
        # e) Log results to file
        results_df = pd.DataFrame([current_pos_results])
        results_df.to_csv(OUTPUT_LOG_PATH, mode='a', header=not header_written, index=False)
        header_written = True # Ensure header is only written once

        all_results.append(current_pos_results)

    # --- 4. Cleanup ---
    print("\nClosing engines...")
    oracle_engine.quit()
    for engine in benchmark_engines.values():
        engine.quit()

    # --- 5. Final Report and Graph ---
    print("Generating final report and graph...")
    
    # Prepare data for the final plot
    final_avg_scores = []
    for name, totals in running_totals.items():
        if totals['count'] > 0:
            avg_score = totals['score_sum'] / totals['count']
            if name == 'player':
                final_avg_scores.append({'name': PLAYER_NAME_IN_PGN, 'type': 'Player', 'avg_score': avg_score})
            else:
                 rating = benchmark_engine_info[benchmark_engine_info['engine_name'] == name].iloc[0]['rating']
                 final_avg_scores.append({'name': name, 'type': 'Engine', 'avg_score': avg_score, 'rating': rating})
    
    plot_df = pd.DataFrame(final_avg_scores)
    engine_plot_df = plot_df[plot_df['type'] == 'Engine'].copy()
    player_plot_df = plot_df[plot_df['type'] == 'Player'].copy()

    # Final regression for the plot
    if len(engine_plot_df) >= 2 and not player_plot_df.empty:
        slope, intercept, r_value, _, _ = linregress(engine_plot_df['avg_score'], engine_plot_df['rating'])
        r_squared = r_value ** 2
        final_player_rating = (slope * player_plot_df.iloc[0]['avg_score']) + intercept
        
        print(f"\n--- Final Results ---")
        print(f"Analysis complete over {running_totals['player']['count']} positions.")
        print(f"Player's Final Average Score: {player_plot_df.iloc[0]['avg_score']:.4f}")
        print(f"Final Estimated Rating for {PLAYER_NAME_IN_PGN}: {final_player_rating:.0f}")
        print(f"Benchmark R-squared: {r_squared:.4f}")

        # Generate Graph
        plt.figure(figsize=(12, 8))
        sns.regplot(x='rating', y='avg_score', data=engine_plot_df, ci=None, line_kws={'color':'red', 'linestyle':'--'}, label='Engine Trend')
        
        # Plot benchmark engines
        for i, row in engine_plot_df.iterrows():
            plt.scatter(row['rating'], row['avg_score'], s=80)
            plt.text(row['rating'] + 10, row['avg_score'], row['name'], fontsize=9)

        # Plot player
        plt.scatter(final_player_rating, player_plot_df.iloc[0]['avg_score'], color='gold', s=200, edgecolor='black', zorder=5, label=f'You ({PLAYER_NAME_IN_PGN})')
        plt.text(final_player_rating + 10, player_plot_df.iloc[0]['avg_score'], 'You', fontsize=11, weight='bold')

        plt.title(f"Final Performance Analysis vs. Engine Rating\nMethod: {CHOSEN_METHOD['name']}", fontsize=16)
        plt.xlabel("Engine Rating", fontsize=12)
        plt.ylabel("Average Hit Score", fontsize=12)
        plt.legend()
        plt.grid(True)
        
        plt.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$\nFinal Est. Rating: {final_player_rating:.0f}', 
                 transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', 
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
                 
        plt.savefig(OUTPUT_GRAPH_PATH)
        print(f"\nGraph saved to: {OUTPUT_GRAPH_PATH}")

    else:
        print("Not enough benchmark data to create a final rating estimate or graph.")

    print("--- Script Finished ---")


if __name__ == "__main__":
    main()
