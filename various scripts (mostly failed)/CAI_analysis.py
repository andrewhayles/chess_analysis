import chess
import chess.pgn
import chess.engine
import pandas as pd
import time
import math
import sys
import os

# ==============================================================================
# --- Configuration ---
# ==============================================================================
# You can adjust these parameters to control the analysis.

# --- Run Control ---
# The maximum number of hours the script should run.
HOURS_TO_RUN = 1.0
# The move number to start the analysis from (e.g., 10 means start at move 10).
# This is useful for skipping the opening book phase of games.
START_MOVE = 10
# The number of positions to analyze for each engine. The script will stop
# once this number is reached for all target engines, or the time limit expires.
NUM_POSITIONS_PER_ENGINE = 50

# --- File Paths ---
# Path to the Stockfish executable.
# PLEASE UPDATE THIS PATH to where you have Stockfish on your system.
# On Windows, it might be "C:/path/to/stockfish.exe"
# On Linux/macOS, it might be "./stockfish"
STOCKFISH_PATH = "stockfish"
# Path to the PGN file containing the games to be analyzed.
PGN_FILE_PATH = "TCEC.txt"
# Path to the CSV file listing the engines to be analyzed.
# The CSV should have a column named 'engine_name'.
ENGINES_CSV_PATH = "real_engines.csv"

# --- CAI Metric Tuning Parameters ---
# These values are based on the "CAI description.txt" file.
# 'k' parameter for the Win Probability (WP) calculation.
K_WP = 0.004
# 'k3' parameter for the Criticality calculation.
K3_CRITICALITY = 0.003
# 'w' weighting factor for amplifying error by criticality.
W_IMPACT = 1.0
# Analysis depth for the oracle engine (Stockfish). Higher values are more accurate but slower.
ANALYSIS_DEPTH = 20

# ==============================================================================
# --- Helper Functions for CAI Calculation ---
# ==============================================================================

def cp_to_wp(score_obj):
    """
    Converts a chess.engine.PovScore object to Win Probability (0-100)
    from the perspective of the current player.
    """
    cp = score_obj.score(mate_score=32000)
    if cp is None:
        # Handle cases like tablebase positions without CP scores
        return 50.0

    # The formula from the description is a variation of the tanh function.
    # WP = 50 + 50 * tanh(k * cp / 2)
    try:
        return 50.0 + 50.0 * math.tanh(K_WP * cp / 2.0)
    except OverflowError:
        return 100.0 if cp > 0 else 0.0

def calculate_cai_metrics(engine, board, actual_move):
    """
    Calculates the CAI (Context-Aware Impact) score for a single move.

    Args:
        engine: The analysis engine (Stockfish).
        board: The chess board state *before* the move is made.
        actual_move: The move that was actually played in the game.

    Returns:
        The calculated CAI score, or None if analysis fails.
    """
    try:
        # 1. Analyze the position BEFORE the move to get the top 2 moves and eval.
        analysis_before = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH), multipv=2)

        if len(analysis_before) < 2:
            # Not enough info to calculate criticality, skip this position.
            return None

        # Get scores from the perspective of the player whose turn it is.
        pov_score_best = analysis_before[0]['score'].pov(board.turn)
        pov_score_second_best = analysis_before[1]['score'].pov(board.turn)

        # 2. Make the actual move and analyze the position AFTER.
        board.push(actual_move)
        info_after = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
        # The score is now from the other player's perspective.
        # We flip it back to the perspective of the player who just moved.
        pov_score_after = info_after['score'].pov(not board.turn)
        board.pop()  # Revert the move for the next iteration.

        # 3. Convert centipawn evaluations to Win Probability (WP).
        wp_before_best_move = cp_to_wp(pov_score_best)
        wp_after_actual_move = cp_to_wp(pov_score_after)

        # 4. Calculate Win Probability Loss (WPLoss).
        # This is the drop in win probability from the player's perspective.
        wp_loss = wp_before_best_move - wp_after_actual_move

        # 5. Calculate Criticality.
        # Delta (Δ) is the absolute difference in evaluation between the top two moves.
        cp1 = pov_score_best.score(mate_score=32000)
        cp2 = pov_score_second_best.score(mate_score=32000)
        delta = abs(cp1 - cp2) if cp1 is not None and cp2 is not None else 0

        # The formula given is equivalent to tanh( (k3 * delta) / 2 ).
        # This maps the evaluation difference to a [0, 1) range.
        criticality = math.tanh(K3_CRITICALITY * delta / 2.0)

        # 6. Calculate the final Impact score.
        impact = wp_loss * (1 + W_IMPACT * criticality)
        
        # The description mentions scaling this to [-100, 100].
        # A simple way is to divide by 2, as the max impact is ~200.
        # However, for analysis, the raw impact score can be more informative.
        # We will return the raw impact and average that.
        return impact

    except chess.engine.EngineTerminatedError:
        print("Stockfish engine terminated unexpectedly. Exiting.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during CAI calculation: {e}", file=sys.stderr)
        return None

# ==============================================================================
# --- Main Analysis Function ---
# ==============================================================================

def analyze_pgn_file():
    """
    Main function to orchestrate the PGN analysis.
    """
    print("--- Starting Chess Engine CAI Analysis ---")
    
    # --- Initialization ---
    start_time = time.time()
    end_time = start_time + HOURS_TO_RUN * 3600

    if not os.path.exists(STOCKFISH_PATH):
        print(f"Error: Stockfish executable not found at '{STOCKFISH_PATH}'")
        print("Please update the STOCKFISH_PATH variable in the script.")
        return

    if not os.path.exists(PGN_FILE_PATH):
        print(f"Error: PGN file not found at '{PGN_FILE_PATH}'")
        return

    if not os.path.exists(ENGINES_CSV_PATH):
        print(f"Error: Engines CSV file not found at '{ENGINES_CSV_PATH}'")
        return

    try:
        print("Initializing Stockfish engine...")
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print("Stockfish engine initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Stockfish engine: {e}", file=sys.stderr)
        return

    try:
        df = pd.read_csv(ENGINES_CSV_PATH)
        engines_to_analyze = set(df['engine_name'].tolist())
        print(f"Loaded {len(engines_to_analyze)} engines to analyze: {', '.join(engines_to_analyze)}")
    except Exception as e:
        print(f"Failed to read or parse '{ENGINES_CSV_PATH}': {e}", file=sys.stderr)
        engine.quit()
        return

    engine_scores = {name: [] for name in engines_to_analyze}
    positions_analyzed = {name: 0 for name in engines_to_analyze}
    games_processed = 0

    # --- Game Processing Loop ---
    print(f"\nStarting analysis. Will run for up to {HOURS_TO_RUN} hours or {NUM_POSITIONS_PER_ENGINE} positions per engine.")
    print("-" * 50)

    with open(PGN_FILE_PATH) as pgn:
        while True:
            # Check for termination conditions
            if time.time() > end_time:
                print("\nTime limit reached. Halting analysis.")
                break

            all_done = all(positions_analyzed.get(name, 0) >= NUM_POSITIONS_PER_ENGINE for name in engines_to_analyze)
            if all_done:
                print("\nRequired number of positions analyzed for all engines. Halting analysis.")
                break

            try:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    print("\nEnd of PGN file reached.")
                    break
            except Exception as e:
                print(f"\nError reading game. Skipping. Error: {e}", file=sys.stderr)
                continue

            games_processed += 1
            white_player = game.headers.get("White", "Unknown")
            black_player = game.headers.get("Black", "Unknown")

            # Process only if at least one player is in our target list
            if not (white_player in engines_to_analyze or black_player in engines_to_analyze):
                continue
            
            print(f"\nProcessing Game {games_processed}: {white_player} vs {black_player}")
            board = game.board()
            
            for i, move in enumerate(game.mainline_moves()):
                move_number = (i // 2) + 1
                if move_number < START_MOVE:
                    board.push(move)
                    continue

                player_to_move = white_player if board.turn() == chess.WHITE else black_player

                if player_to_move in engines_to_analyze and positions_analyzed.get(player_to_move, 0) < NUM_POSITIONS_PER_ENGINE:
                    board_before_move = board.copy()
                    cai_score = calculate_cai_metrics(engine, board_before_move, move)
                    
                    if cai_score is not None:
                        engine_scores[player_to_move].append(cai_score)
                        positions_analyzed[player_to_move] += 1
                        sys.stdout.write(
                            f"\rAnalyzed move {move_number} for {player_to_move}. "
                            f"Positions: {positions_analyzed[player_to_move]}/{NUM_POSITIONS_PER_ENGINE}. "
                            f"Last CAI: {cai_score:.4f}"
                        )
                        sys.stdout.flush()

                board.push(move)
                
                # Check time limit inside the inner loop for responsiveness
                if time.time() > end_time:
                    break
            if time.time() > end_time:
                break


    # --- Final Results ---
    print("\n\n" + "=" * 50)
    print("--- Analysis Complete ---")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
    print(f"Total games processed: {games_processed}")
    print("=" * 50)

    results = []
    for engine_name in engines_to_analyze:
        scores = engine_scores[engine_name]
        if scores:
            average_cai = sum(scores) / len(scores)
            results.append((engine_name, len(scores), average_cai))
        else:
            results.append((engine_name, 0, 0))
    
    # Sort results by average CAI (lower is better)
    results.sort(key=lambda x: x[2])

    print("\n--- Average CAI Scores (Lower is Better) ---")
    print("-" * 55)
    print(f"{'Engine':<25} | {'Positions Analyzed':<20} | {'Average CAI':<15}")
    print("-" * 55)
    for name, count, avg_cai in results:
        print(f"{name:<25} | {count:<20} | {avg_cai:<15.4f}")
    print("-" * 55)

    # --- Cleanup ---
    engine.quit()
    print("Engine closed. Script finished.")


if __name__ == "__main__":
    # Ensure the Stockfish path is correct before running
    if not os.path.exists(STOCKFISH_PATH) and not os.path.exists(STOCKFISH_PATH + ".exe"):
         print(f"ERROR: Stockfish engine not found at '{STOCKFISH_PATH}'.")
         print("Please download a Stockfish executable and place it in the same directory as this script,")
         print("or update the STOCKFISH_PATH variable to its correct location.")
    else:
        analyze_pgn_file()
