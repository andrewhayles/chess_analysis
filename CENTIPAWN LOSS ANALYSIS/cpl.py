# -*- coding: utf-8 -*-
"""
Stockfish PGN Analyzer

This script analyzes a PGN file using a local Stockfish chess engine to calculate
the centipawn loss (CPL) for each move. It then outputs a new PGN file with
the CPL values added as comments.

Prerequisites:
1. Python 3.6+
2. The `python-chess` library. Install it using pip:
   pip install python-chess
3. A local copy of the Stockfish engine executable. You can download it from
   the official website: https://stockfishchess.org/download/

Usage:
Run the script from your terminal. You must provide the path to your
Stockfish executable and the PGN file you want to analyze.

Example:
python pgn_stockfish_analyzer.py "C:\\Users\\User\\Downloads\\stockfish\\stockfish.exe" "my_games.pgn"

The script will generate a new file named `my_games_analyzed.pgn` in the same directory.
"""
import chess
import chess.pgn
import chess.engine
import argparse
import os
import sys

def get_stockfish_path():
    """Tries to find the Stockfish executable automatically."""
    if sys.platform == "win32":
        # Look for .exe in common locations or PATH
        for path in os.environ["PATH"].split(os.pathsep):
            exe_path = os.path.join(path, "stockfish.exe")
            if os.path.exists(exe_path):
                return exe_path
        # A common default install location
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        common_path = os.path.join(program_files, "Stockfish", "stockfish.exe")
        if os.path.exists(common_path):
            return common_path
    else: # Linux or MacOS
        for path in os.environ["PATH"].split(os.pathsep):
            exe_path = os.path.join(path, "stockfish")
            if os.path.exists(exe_path) and os.access(exe_path, os.X_OK):
                return exe_path
    return None


def analyze_game_and_annotate(game, engine, analysis_limit):
    """
    Analyzes a single game, calculates CPL for each move, and adds it as a comment.
    Returns the modified game object.
    """
    board = game.board()
    # The game node iterator will be consumed, so we work on a new game object
    new_game = chess.pgn.Game()
    new_game.headers.update(game.headers)
    node = new_game

    # Get initial position evaluation
    try:
        info = engine.analyse(board, limit=analysis_limit)
        score_before = info["score"].white()
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
        print(f"\nEngine error analyzing initial position: {e}", file=sys.stderr)
        return None # Cannot analyze this game

    for move in game.mainline_moves():
        # Analyze the position *after* the move has been made
        board.push(move)
        try:
            info = engine.analyse(board, limit=analysis_limit)
            score_after = info["score"].white()
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
            print(f"\nEngine error during game: {e}", file=sys.stderr)
            # Stop analysis for this game but keep moves made so far
            break

        # Calculate centipawn loss
        cpl = 0
        if score_before.is_mate() or score_after.is_mate():
             # CPL is tricky with mate scores, often better to assign a high value or 0
             # For simplicity, we'll assign 0 if the evaluation is consistent.
             if score_before.mate() is not None and score_after.mate() is not None:
                 if (score_before.mate() > 0 and score_after.mate() > 0) or \
                    (score_before.mate() < 0 and score_after.mate() < 0):
                     cpl = 0
                 else:
                     cpl = 350 # A blunder that turns a win into a loss
             else:
                 cpl = 350 # Moving from a non-mate to a mate score is a big change
        else:
            # All scores are from White's perspective
            eval_before = score_before.score()
            eval_after = score_after.score()
            
            if board.turn == chess.BLACK:  # White just moved
                cpl = eval_before - eval_after
            else:  # Black just moved
                cpl = eval_after - eval_before
        
        cpl = max(0, int(cpl)) # CPL can't be negative

        # Add the move and the CPL comment to our new game tree
        node = node.add_variation(move)
        node.comment += f" [%cpl {cpl}]"
        
        # The score for the new position becomes the score for the next iteration
        score_before = score_after
        
    return new_game


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyzes a PGN file with Stockfish to add centipawn loss (CPL) comments.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example:\npython pgn_stockfish_analyzer.py --stockfish ./stockfish --pgn my_games.pgn"
    )
    
    # Try to find stockfish automatically
    default_stockfish_path = get_stockfish_path()

    parser.add_argument(
        "--stockfish",
        dest="stockfish_path",
        required=default_stockfish_path is None,
        default=default_stockfish_path,
        help="Path to the Stockfish executable. Required if not found automatically."
    )
    parser.add_argument(
        "--pgn",
        dest="pgn_path",
        required=True,
        help="Path to the input PGN file."
    )
    parser.add_argument(
        "--time",
        dest="analysis_time",
        type=float,
        default=0.1,
        help="Time limit in seconds for engine analysis per move (e.g., 0.5 for half a second)."
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Path for the output PGN file. Defaults to '[input_name]_analyzed.pgn'."
    )

    args = parser.parse_args()

    # --- Validate paths ---
    if not os.path.exists(args.stockfish_path):
        print(f"Error: Stockfish executable not found at '{args.stockfish_path}'", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(args.pgn_path):
        print(f"Error: PGN file not found at '{args.pgn_path}'", file=sys.stderr)
        sys.exit(1)

    # --- Setup output file path ---
    output_path = args.output_path
    if not output_path:
        base, _ = os.path.splitext(args.pgn_path)
        output_path = f"{base}_analyzed.pgn"

    print(f"Using Stockfish: {args.stockfish_path}")
    print(f"Analyzing PGN: {args.pgn_path}")
    print(f"Analysis time per move: {args.analysis_time}s")
    print(f"Output will be saved to: {output_path}")
    
    engine = None
    try:
        # --- Initialize Stockfish Engine ---
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        analysis_limit = chess.engine.Limit(time=args.analysis_time)
        
        game_count = 0
        games_analyzed = 0
        
        # --- Process PGN File ---
        with open(args.pgn_path) as pgn, open(output_path, "w", encoding="utf-8") as out_pgn:
            while True:
                game_count += 1
                try:
                    game = chess.pgn.read_game(pgn)
                except (ValueError, RuntimeError):
                    # Malformed PGN can cause errors, skip to next game
                    print(f"\nWarning: Could not parse game #{game_count}. It may be malformed.", file=sys.stderr)
                    continue

                if game is None:
                    break # End of file
                
                print(f"\rAnalyzing game {game_count}...", end="")
                sys.stdout.flush()

                analyzed_game = analyze_game_and_annotate(game, engine, analysis_limit)
                
                if analyzed_game:
                    exporter = chess.pgn.FileExporter(out_pgn)
                    analyzed_game.accept(exporter)
                    games_analyzed += 1

        print(f"\n\nAnalysis complete.")
        print(f"Successfully analyzed and saved {games_analyzed} games to '{output_path}'.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if engine:
            engine.quit()

if __name__ == "__main__":
    main()
