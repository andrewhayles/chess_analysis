import json
import os

# ==============================================================================
# --- Configuration ---
# ==============================================================================
# File path for the oracle cache generated with a lower depth (e.g., 22)
FILE_DEPTH_22 = 'oracle_cache_old.json' 
# File path for the oracle cache generated with a higher depth (e.g., 30)
FILE_DEPTH_30 = 'oracle_cache_top3.json'

# ==============================================================================
# --- Main Comparison Logic ---
# ==============================================================================

def load_json_cache(file_path):
    """Loads a JSON cache from a file, handling potential errors."""
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None
    
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"[ERROR] Could not decode JSON from {file_path}. The file might be corrupted.")
            return None

def compare_oracle_caches(file1, file2):
    """
    Compares two oracle cache files to see how often their top move recommendations agree.
    """
    print("--- Starting Oracle Cache Comparison ---")
    
    # Load the two cache files
    cache_d22 = load_json_cache(file1)
    cache_d30 = load_json_cache(file2)

    # Exit if either file failed to load
    if cache_d22 is None or cache_d30 is None:
        print("[EXIT] Cannot proceed without both cache files.")
        return

    print(f"Found {len(cache_d22)} positions in '{file1}' (Depth 22).")
    print(f"Found {len(cache_d30)} positions in '{file2}' (Depth 30).")

    # Find the set of common FENs (positions) between the two files
    common_fens = set(cache_d22.keys()) & set(cache_d30.keys())
    
    if not common_fens:
        print("\n[RESULT] No common positions found between the two files. Cannot perform comparison.")
        return

    print(f"\nFound {len(common_fens)} common positions to compare.")

    # Initialize counters
    agreement_count = 0
    disagreement_count = 0
    
    # --- Data Structure Normalization ---
    # The script needs to handle two different JSON structures you've used.
    # Old structure: {fen: ["move1", "move2", ...]}
    # New structure: {fen: {"oracle_moves": ["move1", ...], "player_move": "..."}}
    
    for fen in common_fens:
        # Get moves from the first file (Depth 22)
        moves_d22_data = cache_d22[fen]
        if isinstance(moves_d22_data, list) and moves_d22_data:
            top_move_d22 = moves_d22_data[0]
        else:
            # Handle cases where the format might be different or the list is empty
            continue 

        # Get moves from the second file (Depth 30)
        moves_d30_data = cache_d30[fen]
        if isinstance(moves_d30_data, dict) and "oracle_moves" in moves_d30_data and moves_d30_data["oracle_moves"]:
            top_move_d30 = moves_d30_data["oracle_moves"][0]
        elif isinstance(moves_d30_data, list) and moves_d30_data:
             top_move_d30 = moves_d30_data[0]
        else:
            # Skip if the structure is unexpected or there are no moves
            continue
            
        # Compare the top recommended move
        if top_move_d22 == top_move_d30:
            agreement_count += 1
        else:
            disagreement_count += 1

    # --- Calculate and Print Results ---
    total_compared = agreement_count + disagreement_count
    if total_compared > 0:
        agreement_percentage = (agreement_count / total_compared) * 100
        print("\n--- Comparison Results ---")
        print(f"Total Positions Compared: {total_compared}")
        print(f"Top Move Agreements:      {agreement_count}")
        print(f"Top Move Disagreements:   {disagreement_count}")
        print(f"Agreement Percentage:     {agreement_percentage:.2f}%")
    else:
        print("\n[RESULT] Could not compare any positions due to data format issues.")

if __name__ == "__main__":
    compare_oracle_caches(FILE_DEPTH_22, FILE_DEPTH_30)
