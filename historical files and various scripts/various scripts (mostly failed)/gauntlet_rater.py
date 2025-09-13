# gauntlet_rater.py
# A script to automatically find all engines in a folder, run a gauntlet tournament
# against a reference engine, estimate their Elo, and create the real_engines.csv file.

import chess
import chess.engine
import chess.pgn
from pathlib import Path
import logging
import math

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
ENGINES_DIR = PROJECT_FOLDER / "engines"

# This is the known, strong engine that all others will be tested against.
# Make sure this path is correct.
REFERENCE_ENGINE_PATH = PROJECT_FOLDER / "engines" / "stockfish - 3644" / "stockfish-windows-x86-64-sse41-popcnt.exe"
REFERENCE_ENGINE_RATING = 3644  # The known rating of our reference engine.

# Tournament settings
GAMES_PER_MATCHUP = 10  # Number of games each engine plays against the reference. (5 as white, 5 as black)
TIME_LIMIT_PER_MOVE = 0.5  # Time in seconds for each move.

# --- SCRIPT SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if not REFERENCE_ENGINE_PATH.is_file():
    logging.error(f"FATAL: Reference engine not found at '{REFERENCE_ENGINE_PATH}'")
    logging.error("Please ensure the compatible Stockfish engine is downloaded and the path is correct.")
    exit()

# --- CORE FUNCTIONS ---

def find_all_engines(directory: Path, exclude_path: Path):
    """Finds all .exe files in a directory, excluding any with the same name as the reference engine."""
    logging.info(f"Scanning for engines in: {directory}")
    all_executables = [p for p in directory.glob('**/*.exe')]
    
    reference_filename = exclude_path.name
    
    # Filter out any engine that has the same filename as the reference engine.
    # This prevents testing copies of the reference engine.
    filtered_executables = [p for p in all_executables if p.name != reference_filename]
    
    logging.info(f"Found {len(filtered_executables)} engines to test.")
    return filtered_executables

def play_game(white_engine_path: Path, black_engine_path: Path) -> str:
    """
    Plays a single game between two engines and returns the result.
    Returns '1-0', '0-1', or '1/2-1/2'.
    """
    try:
        with chess.engine.SimpleEngine.popen_uci(white_engine_path) as white_engine, \
             chess.engine.SimpleEngine.popen_uci(black_engine_path) as black_engine:
            
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["White"] = white_engine_path.stem
            game.headers["Black"] = black_engine_path.stem
            node = game

            while not board.is_game_over(claim_draw=True):
                if board.turn == chess.WHITE:
                    result = white_engine.play(board, chess.engine.Limit(time=TIME_LIMIT_PER_MOVE))
                else:
                    result = black_engine.play(board, chess.engine.Limit(time=TIME_LIMIT_PER_MOVE))
                
                if result.move is None: # Engine might fail to produce a move
                    break
                    
                node = node.add_variation(result.move)
                board.push(result.move)

            game.headers["Result"] = board.result(claim_draw=True)
            return game.headers["Result"]

    except Exception as e:
        logging.error(f"Error during game between {white_engine_path.name} and {black_engine_path.name}: {e}")
        return "*" # Return unknown result on error

def calculate_elo(score: float, num_games: int, opponent_rating: int) -> int:
    """
    Calculates the estimated Elo rating based on the score against a known opponent.
    """
    if num_games == 0 or score < 0 or score > num_games:
        return 0 # Invalid score

    # Avoid division by zero if score is 0 or num_games
    if score == 0: score = 0.01
    if score == num_games: score = num_games - 0.01

    expected_score_percent = score / num_games
    elo_diff = -400 * math.log10(1 / expected_score_percent - 1)
    
    return int(opponent_rating + elo_diff)

def main():
    """
    Main workflow to find, test, and rate all engines.
    """
    # Step 1: Find all engines to be tested
    engines_to_test = find_all_engines(ENGINES_DIR, REFERENCE_ENGINE_PATH)
    if not engines_to_test:
        logging.warning("No other engines found to test. Exiting.")
        return

    print("\n--- Engine Gauntlet Rater ---")
    print(f"The following {len(engines_to_test)} engines will be tested against {REFERENCE_ENGINE_PATH.name}:")
    for i, engine_path in enumerate(engines_to_test):
        print(f"  {i+1}. {engine_path.relative_to(PROJECT_FOLDER)}")
    
    confirm = input("\nIs this list correct? Press Enter to begin the tournament, or 'n' to cancel: ")
    if confirm.lower() == 'n':
        print("Tournament cancelled.")
        return

    # Step 2: Run the gauntlet tournament
    engine_ratings = {}
    for engine_path in engines_to_test:
        logging.info(f"\n--- Matchup: {engine_path.name} vs. {REFERENCE_ENGINE_PATH.name} ---")
        score = 0.0
        
        # Play games as White
        logging.info(f"Playing {GAMES_PER_MATCHUP // 2} games as White...")
        for i in range(GAMES_PER_MATCHUP // 2):
            result = play_game(engine_path, REFERENCE_ENGINE_PATH)
            if result == '1-0': score += 1.0
            elif result == '1/2-1/2': score += 0.5
            logging.info(f"  Game {i+1} (W): {result}. Current score: {score}")

        # Play games as Black
        logging.info(f"Playing {GAMES_PER_MATCHUP - (GAMES_PER_MATCHUP // 2)} games as Black...")
        for i in range(GAMES_PER_MATCHUP - (GAMES_PER_MATCHUP // 2)):
            result = play_game(REFERENCE_ENGINE_PATH, engine_path)
            if result == '0-1': score += 1.0
            elif result == '1/2-1/2': score += 0.5
            logging.info(f"  Game {i+1} (B): {result}. Current score: {score}")
            
        estimated_rating = calculate_elo(score, GAMES_PER_MATCHUP, REFERENCE_ENGINE_RATING)
        engine_ratings[str(engine_path)] = estimated_rating
        logging.info(f"Match complete for {engine_path.name}. Score: {score}/{GAMES_PER_MATCHUP}. Estimated Rating: {estimated_rating}")

    # Step 3: Write the results to real_engines.csv
    csv_path = PROJECT_FOLDER / "real_engines.csv"
    logging.info(f"\n--- Writing results to {csv_path} ---")
    
    # Add the reference engine to the list first
    all_rated_engines = {str(REFERENCE_ENGINE_PATH): REFERENCE_ENGINE_RATING}
    all_rated_engines.update(engine_ratings)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("path,rating\n")
        for path, rating in all_rated_engines.items():
            f.write(f'"{path}",{rating}\n')
            print(f"  - Wrote: {Path(path).name}, {rating}")
            
    print("\n--- Gauntlet Tournament Finished! ---")
    print(f"The file '{csv_path.name}' has been created with all engine ratings.")
    print("You can now use this file with the main `chess_analyzer_final.py` script by setting USE_REAL_ENGINES = True.")

if __name__ == "__main__":
    main()
