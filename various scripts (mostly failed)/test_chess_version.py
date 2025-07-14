# test_chess_version.py
import chess
import sys

print(f"--- Chess Library Test ---")
print(f"Python Executable: {sys.executable}")
print(f"Found chess library file at: {chess.__file__}")
print(f"Library Version: {chess.__version__}")

try:
    # Create a centipawn score object, just like in the main script
    score_object = chess.engine.Cp(123)

    print("\nAttempting to call .score(mate_ok=True)...")
    # This is the line that causes the error
    score_object.score(mate_ok=True)

    print("\nSUCCESS! The installed chess library is up-to-date and working correctly.")

except TypeError as e:
    print(f"\nFAILURE! The error is still present.")
    print(f"Error message: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

print("\n--- Test Complete ---")