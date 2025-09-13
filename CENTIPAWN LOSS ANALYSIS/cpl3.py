# -*- coding: utf-8 -*-
"""
Stockfish PGN Analyzer (Overnight Version with ACPL Report)

This script can operate in two modes:

1. Full Analysis Mode:
   Analyzes a PGN file using a local Stockfish chess engine to calculate
   the centipawn loss (CPL) for each move. It's designed for long-running,
   deep analysis by distributing a total specified duration across all moves.
   After analysis, it saves an annotated PGN and generates an ACPL report
   for a specified player.

2. Report-Only Mode:
   If a PGN file has already been analyzed (i.e., it contains '[%cpl ...]'
   comments), this mode skips the engine analysis and directly generates
   the ACPL report for the specified player.

Prerequisites:
- Python 3.6+
- The `python-chess` library (`pip install python-chess`)
- A local Stockfish engine (only for Full Analysis Mode)

Usage:
# Full Analysis Example (9-hour analysis for player 'Player123'):
python cpl2.py --stockfish "C:\\stockfish\\stockfish.exe" --pgn "my_games.pgn" --duration 9 --player "Player123"

# Report-Only Example (for an already analyzed PGN):
python cpl2.py --pgn "my_games_analyzed.pgn" --player "Player123" --report-only
"""
import chess
import chess.pgn
import chess.engine
import argparse
import os
import sys
import time
import re

def get_stockfish_path():
    """Tries to find the Stockfish executable automatically."""
    if sys.platform == "win32":
        for path in os.environ["PATH"].split(os.pathsep):
            exe_path = os.path.join(path, "stockfish.exe")
            if os.path.exists(exe_path):
                return exe_path
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
    Returns the modified game object and a list of CPL values for each move.
    """
    board = game.board()
    new_game = chess.pgn.Game()
    new_game.headers.update(game.headers)
    node = new_game
    cpl_data = []

    try:
        info = engine.analyse(board, limit=analysis_limit)
        score_before = info["score"].white()
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
        print(f"\nEngine error analyzing initial position: {e}", file=sys.stderr)
        return None, []

    for move in game.mainline_moves():
        board.push(move)
        try:
            info = engine.analyse(board, limit=analysis_limit)
            score_after = info["score"].white()
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
            print(f"\nEngine error during game: {e}", file=sys.stderr)
            break

        cpl = 0
        if score_before.is_mate() or score_after.is_mate():
             if score_before.mate() is not None and score_after.mate() is not None:
                 if (score_before.mate() > 0 and score_after.mate() > 0) or \
                    (score_before.mate() < 0 and score_after.mate() < 0):
                     cpl = 0
                 else:
                     cpl = 350
             else:
                 cpl = 350
        else:
            eval_before = score_before.score()
            eval_after = score_after.score()
            
            if board.turn == chess.BLACK:  # White just moved
                cpl = eval_before - eval_after
            else:  # Black just moved
                cpl = eval_after - eval_before
        
        cpl = max(0, int(cpl))
        cpl_data.append(cpl)

        node = node.add_variation(move)
        node.comment += f" [%cpl {cpl}]"
        
        score_before = score_after
        
    return new_game, cpl_data


def generate_acpl_report(player_name, acpl, total_moves, report_path):
    """Generates a text file with the ACPL analysis."""
    
    skill_level = "Unknown"
    if acpl < 10:
        skill_level = "Grandmaster / Engine Level"
    elif acpl < 20:
        skill_level = "Master Level"
    elif acpl < 35:
        skill_level = "Expert Level"
    elif acpl < 50:
        skill_level = "Advanced Player"
    elif acpl < 75:
        skill_level = "Intermediate Player"
    else:
        skill_level = "Beginner / Casual Player"

    content = f"""
ACPL (Average Centipawn Loss) Report for: {player_name}
=====================================================

Calculated ACPL: {acpl:.2f}
Analyzed Moves: {total_moves}

Estimated Skill Level: {skill_level}

---
What is ACPL?
---
Average Centipawn Loss (ACPL) is a metric used to measure the accuracy of chess moves.
A "centipawn" is 1/100th of a pawn. ACPL represents the average number of centipawns
you lost per move compared to the engine's best move. A lower ACPL means your moves
were closer to the computer's top choices, indicating higher accuracy.

---
General ACPL Ranges (Estimates):
---
-   0-10:   Grandmaster Level: World-class precision, very close to perfect play.
-  10-20:   Master Level: Extremely strong and consistent play with very few inaccuracies.
-  20-35:   Expert Level: A very strong player who understands complex positions well.
-  35-50:   Advanced Player: Solid tactical and strategic understanding.
-  50-75:   Intermediate Player: Good grasp of fundamentals but may miss complex tactics.
-  75+:     Beginner / Casual: Still learning core principles, prone to frequent blunders.

Note: These ranges are approximate and can be affected by the complexity of the games,
time control, and the depth of the analysis engine.
"""
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Successfully generated ACPL report at '{report_path}'.")
    except IOError as e:
        print(f"\nError writing report file: {e}", file=sys.stderr)


def calculate_acpl_from_analyzed_pgn(pgn_path, player_name):
    """
    Calculates ACPL for a player by reading [%cpl] comments from an analyzed PGN.
    """
    total_player_cpl = 0
    total_player_moves = 0
    cpl_pattern = re.compile(r"\[%cpl (\d+)\]")
    game_count = 0

    print("Step 1: Parsing CPL data from analyzed PGN...")
    try:
        with open(pgn_path, encoding='utf-8') as pgn:
            while True:
                game_count += 1
                print(f"\rProcessing game {game_count}...", end="")
                sys.stdout.flush()

                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                
                white_player = game.headers.get("White", "").strip()
                black_player = game.headers.get("Black", "").strip()
                
                is_white = white_player.lower() == player_name.lower()
                is_black = black_player.lower() == player_name.lower()

                if not is_white and not is_black:
                    continue

                node = game
                ply_count = 0
                while node.variations:
                    node = node.variation(0)
                    match = cpl_pattern.search(node.comment)
                    if match:
                        cpl = int(match.group(1))
                        
                        if ply_count % 2 == 0 and is_white:
                            total_player_cpl += cpl
                            total_player_moves += 1
                        elif ply_count % 2 != 0 and is_black:
                            total_player_cpl += cpl
                            total_player_moves += 1
                    
                    ply_count += 1
    except Exception as e:
        print(f"\nError reading PGN during parsing phase: {e}", file=sys.stderr)
        return 0, 0
    
    print("\nParsing complete.")
    return total_player_cpl, total_player_moves


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyzes a PGN file with Stockfish or generates an ACPL report from an already analyzed file.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Full Analysis Example:\n"
            "python cpl2.py --stockfish ./stockfish --pgn games.pgn --duration 9 --player 'YourName'\n\n"
            "Report-Only Example:\n"
            "python cpl2.py --pgn games_analyzed.pgn --player 'YourName' --report-only"
        )
    )
    
    default_stockfish_path = get_stockfish_path()

    parser.add_argument("--pgn", dest="pgn_path", required=True, help="Path to the input PGN file.")
    parser.add_argument("--player", dest="player_name", required=True, help="The username of the player to calculate ACPL for.")
    parser.add_argument("--report", dest="report_path", default="acpl_report.txt", help="Path for the ACPL report file. Defaults to 'acpl_report.txt'.")
    
    # Analysis-specific arguments
    parser.add_argument("--stockfish", dest="stockfish_path", default=default_stockfish_path, help="Path to the Stockfish executable (required for analysis).")
    parser.add_argument("--duration", dest="total_duration_hours", type=float, default=9.0, help="Total desired analysis duration in hours.")
    parser.add_argument("--output", dest="output_path", help="Path for the output PGN file. Defaults to '[input_name]_analyzed.pgn'.")
    parser.add_argument('--report-only', action='store_true', help="Skip analysis and generate ACPL report from an already analyzed PGN.")

    args = parser.parse_args()
    
    if not os.path.exists(args.pgn_path):
        print(f"Error: PGN file not found at '{args.pgn_path}'", file=sys.stderr)
        sys.exit(1)

    if args.report_only:
        # --- REPORT-ONLY MODE ---
        print("\n--- Report-Only Mode ---")
        total_player_cpl, total_player_moves = calculate_acpl_from_analyzed_pgn(args.pgn_path, args.player_name)
        
        print("\nStep 2: Generating ACPL Report...")
        if total_player_moves > 0:
            player_acpl = total_player_cpl / total_player_moves
            print(f"Player '{args.player_name}' found with {total_player_moves} moves.")
            print(f"Average Centipawn Loss (ACPL): {player_acpl:.2f}")
            generate_acpl_report(args.player_name, player_acpl, total_player_moves, args.report_path)
        else:
            print(f"Warning: No moves found for player '{args.player_name}' with CPL data.", file=sys.stderr)
            print("Please check the player name and ensure the PGN file contains '[%cpl ...]' comments.", file=sys.stderr)

    else:
        # --- FULL ANALYSIS MODE ---
        print("\n--- Full Analysis Mode ---")
        if not args.stockfish_path or not os.path.exists(args.stockfish_path):
            print(f"Error: Stockfish executable not found at '{args.stockfish_path}'", file=sys.stderr)
            print("Stockfish path is required for Full Analysis Mode. Use --stockfish or ensure it's in your PATH.", file=sys.stderr)
            sys.exit(1)
        
        start_time = time.time()
        output_path = args.output_path
        if not output_path:
            base, _ = os.path.splitext(args.pgn_path)
            output_path = f"{base}_analyzed.pgn"

        print("Step 1: Counting total moves in PGN file...")
        total_moves = 0
        try:
            with open(args.pgn_path, encoding='utf-8') as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None: break
                    total_moves += len(list(game.mainline_moves()))
        except Exception as e:
            print(f"\nError reading PGN during counting phase: {e}", file=sys.stderr)
            sys.exit(1)

        if total_moves == 0:
            print("No moves found in the PGN file. Nothing to analyze.", file=sys.stderr)
            sys.exit(0)
        
        print(f"Found {total_moves} moves to analyze.")

        total_duration_seconds = args.total_duration_hours * 3600
        time_per_move = total_duration_seconds / total_moves
        
        print("\n--- Analysis Plan ---")
        print(f"Target Player: {args.player_name}")
        print(f"Target duration: {args.total_duration_hours} hours ({total_duration_seconds:.0f} seconds)")
        print(f"Calculated time per move: {time_per_move:.4f} seconds")
        print("---------------------\n")
        print("Step 2: Starting analysis...")
        
        total_player_cpl = 0
        total_player_moves = 0
        engine = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
            analysis_limit = chess.engine.Limit(time=time_per_move)
            
            with open(args.pgn_path, encoding='utf-8') as pgn, open(output_path, "w", encoding="utf-8") as out_pgn:
                game_count = 0
                while True:
                    game_count += 1
                    game = chess.pgn.read_game(pgn)
                    if game is None: break
                    
                    print(f"\rAnalyzing game {game_count}...", end="")
                    sys.stdout.flush()

                    analyzed_game, cpl_data = analyze_game_and_annotate(game, engine, analysis_limit)
                    
                    if cpl_data:
                        white_player = game.headers.get("White", "").strip()
                        black_player = game.headers.get("Black", "").strip()

                        for i, cpl in enumerate(cpl_data):
                            if i % 2 == 0 and white_player.lower() == args.player_name.lower():
                                total_player_cpl += cpl
                                total_player_moves += 1
                            elif i % 2 != 0 and black_player.lower() == args.player_name.lower():
                                total_player_cpl += cpl
                                total_player_moves += 1
                    
                    if analyzed_game:
                        exporter = chess.pgn.FileExporter(out_pgn)
                        analyzed_game.accept(exporter)
            
            actual_duration = time.time() - start_time
            print(f"\n\nAnalysis complete.")
            print(f"Saved analyzed PGN to '{output_path}'.")
            print(f"Total time elapsed: {actual_duration / 3600:.2f} hours.")

            print("\nStep 3: Generating ACPL Report...")
            if total_player_moves > 0:
                player_acpl = total_player_cpl / total_player_moves
                print(f"Player '{args.player_name}' found with {total_player_moves} moves.")
                print(f"Average Centipawn Loss (ACPL): {player_acpl:.2f}")
                generate_acpl_report(args.player_name, player_acpl, total_player_moves, args.report_path)
            else:
                print(f"Warning: No moves found for player '{args.player_name}'. Could not generate ACPL report.", file=sys.stderr)

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        finally:
            if engine:
                engine.quit()

if __name__ == "__main__":
    main()

