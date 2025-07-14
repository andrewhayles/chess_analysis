import chess.pgn
import os

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- PGN File Path ---
# IMPORTANT: This must be the same PGN file used by your main analysis script.
PLAYER_PGN_PATH = "goalgames.pgn" 

# --- FEN to Find ---
# Paste the FEN string from your granular_analysis_log.csv file here.
FEN_TO_FIND = "r2q1rk1/1p2npp1/2ppbn1p/p3p3/3PP3/P1NBPN1P/1PP3P1/R2Q1RK1 b - - 1 13"

# ==============================================================================
# --- Core Logic ---
# ==============================================================================

def find_game_by_fen(pgn_path, fen_string):
    """
    Searches a PGN file for a game containing a specific FEN position.
    """
    print(f"Searching for FEN: {fen_string}")
    print(f"In PGN file: {pgn_path}\n")

    if not os.path.exists(pgn_path):
        print(f"Error: PGN file not found at '{pgn_path}'.")
        return

    found_game = False
    with open(pgn_path, 'r', errors='ignore') as pgn_file:
        game_number = 0
        while True:
            try:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break # End of file
                
                game_number += 1
                board = game.board()

                # Check the starting position as well
                if board.fen() == fen_string:
                    found_game = True
                
                # Iterate through the moves of the game
                if not found_game:
                    for move in game.mainline_moves():
                        board.push(move)
                        if board.fen() == fen_string:
                            found_game = True
                            break # Found it in this game, no need to check further moves
                
                if found_game:
                    print("--- Game Found! ---")
                    print(f"Game Number in PGN: {game_number}")
                    for header, value in game.headers.items():
                        print(f"{header}: {value}")
                    print("-------------------")
                    return # Stop after finding the first match

            except Exception as e:
                # This can happen with malformed games in the PGN, just skip them.
                print(f"Warning: Skipping a game due to a parsing error: {e}")
                continue

    if not found_game:
        print("--- No Match Found ---")
        print("The specified FEN was not found in any game in the PGN file.")


if __name__ == "__main__":
    find_game_by_fen(PLAYER_PGN_PATH, FEN_TO_FIND)
