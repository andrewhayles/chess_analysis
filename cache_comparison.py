import json
import os

# ==============================================================================
# --- Configuration ---
# ==============================================================================

# --- File Paths ---
# Point these to the two oracle cache files you want to compare.
NEW_CACHE_PATH = 'oracle_cache.json'
OLD_CACHE_PATH = 'oracle_cache_old.json'

# ==============================================================================
# --- Core Logic ---
# ==============================================================================

def compare_oracle_caches():
    """
    Loads two oracle cache files, compares the top-rated move for common positions,
    and reports the number of differences.
    """
    print("--- Oracle Cache Comparison Script ---")

    # --- 1. Check for necessary files ---
    if not os.path.exists(NEW_CACHE_PATH):
        print(f"Error: New cache file not found at '{NEW_CACHE_PATH}'.")
        return
    if not os.path.exists(OLD_CACHE_PATH):
        print(f"Error: Old cache file not found at '{OLD_CACHE_PATH}'.")
        return

    # --- 2. Load Data from JSON files ---
    try:
        with open(NEW_CACHE_PATH, 'r') as f:
            new_cache = json.load(f)
        with open(OLD_CACHE_PATH, 'r') as f:
            old_cache = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from one of the files: {e}")
        return
    except Exception as e:
        print(f"An error occurred while reading the files: {e}")
        return

    print(f"Loaded {len(new_cache)} positions from the new cache.")
    print(f"Loaded {len(old_cache)} positions from the old cache.")

    # --- 3. Find Common Positions and Compare ---
    new_fens = set(new_cache.keys())
    old_fens = set(old_cache.keys())
    
    common_fens = new_fens.intersection(old_fens)
    
    if not common_fens:
        print("\nNo common positions found between the two cache files. Cannot compare.")
        return

    print(f"\nFound {len(common_fens)} common positions to compare.")
    
    differences_count = 0
    
    for fen in common_fens:
        # Ensure both entries have at least one move to compare
        if not new_cache[fen] or not old_cache[fen]:
            continue

        new_top_move = new_cache[fen][0]
        old_top_move = old_cache[fen][0]
        
        if new_top_move != old_top_move:
            differences_count += 1
            print("-" * 20)
            print(f"Difference found for FEN: {fen}")
            print(f"  - New Cache Top Move: {new_top_move}")
            print(f"  - Old Cache Top Move: {old_top_move}")
            print("-" * 20)

    # --- 4. Report Final Results ---
    print("\n--- Comparison Complete ---")
    print(f"Total common positions checked: {len(common_fens)}")
    print(f"Number of positions with a different top move: {differences_count}")
    
    if differences_count > 0:
        percentage = (differences_count / len(common_fens)) * 100
        print(f"Percentage of differing top moves: {percentage:.2f}%")
    print("---------------------------\n")


if __name__ == "__main__":
    compare_oracle_caches()
