# interactive_engine_downloader.py
# A semi-automatic script to help you download a curated list of UCI chess engines.
# This version guides you to download manually, then automates the setup.

import requests
import zipfile
import py7zr
from pathlib import Path
import io
import logging
import platform
import warnings
import time
import os

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
ENGINES_DIR = PROJECT_FOLDER / "engines"
DOWNLOADS_DIR = PROJECT_FOLDER / "downloads" # New folder for manual downloads

# --- CURATED ENGINE LIST (STABLE RELEASE PAGES) ---
ENGINE_DATABASE = [
    {"name": "Stockfish 17.1 (Compatible)", "rating": 3644, "url": "https://github.com/official-stockfish/Stockfish/releases/tag/sf_17.1", "notes": "Find and download 'stockfish-windows-x86-64-sse41-popcnt.zip'"},
    {"name": "Koivisto 9.0", "rating": 3515, "url": "https://github.com/Luecx/Koivisto/releases", "notes": "Find and download 'Koivisto-9.0-windows.zip'"},
    {"name": "Seer 2.8.0", "rating": 3490, "url": "https://github.com/GediminasMasaitis/Seer/releases", "notes": "Find and download 'Seer_v2.8.0_Windows.zip'"},
    {"name": "Alexandria 7.0.0", "rating": 3410, "url": "https://github.com/Alexandria-Chess/Alexandria/releases", "notes": "Find and download 'Alexandria-v7.0.0-windows.zip'"},
    {"name": "Arasan 25.1", "rating": 3394, "url": "https://github.com/jdart/arasan/releases", "notes": "Find and download 'arasan-25.1-win.zip'"},
    {"name": "Toga II 4.1.0", "rating": 2850, "url": "https://github.com/Toga-II/Toga-II/releases", "notes": "Find and download 'toga-4.1.0-windows.zip'"},
    {"name": "Amateur 3.10", "rating": 2200, "url": "https://github.com/amateur-chess-engine/amateur/releases", "notes": "Find and download 'amateur-3.10.zip'"},
    {"name": "Tojiko 2.0", "rating": 2000, "url": "https://github.com/jquesada2016/tojiko/releases", "notes": "Find and download 'tojiko-2.0-windows.zip'"},
]


# --- SCRIPT SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ENGINES_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)

def find_executable(extract_path: Path) -> str:
    """Tries to find the most likely executable in the extracted folder."""
    executables = list(extract_path.glob('**/*.exe'))
    if executables:
        return str(executables[0])
    return "NOT_FOUND"

def process_downloaded_file(engine_info):
    """Finds the newest downloaded archive, extracts it, and cleans up."""
    try:
        # Find the most recently downloaded file in the downloads folder
        files = list(DOWNLOADS_DIR.glob('*.zip')) + list(DOWNLOADS_DIR.glob('*.7z'))
        if not files:
            logging.error("No archive file found in the 'downloads' folder. Skipping.")
            return

        latest_file = max(files, key=os.path.getctime)
        logging.info(f"Processing downloaded file: {latest_file.name}")

        engine_name = engine_info['name'].replace(" ", "_").lower()
        extract_path = ENGINES_DIR / engine_name
        extract_path.mkdir(exist_ok=True)
        
        logging.info(f"Extracting to {extract_path}...")
        if latest_file.suffix == '.zip':
            with zipfile.ZipFile(latest_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif latest_file.suffix == '.7z':
            with py7zr.SevenZipFile(latest_file, mode='r') as z_ref:
                z_ref.extractall(path=extract_path)

        logging.info(f"Successfully extracted {engine_name}.")
        
        # Clean up the downloaded archive
        os.remove(latest_file)
        logging.info(f"Removed downloaded archive: {latest_file.name}")

        exe_path = find_executable(extract_path)
        if exe_path != "NOT_FOUND":
            logging.info(f"Found executable: {exe_path}")
            with open(PROJECT_FOLDER / "real_engines.csv", "a") as f:
                f.write(f'"{exe_path}",{engine_info["rating"]}\n')
            logging.info(f"Added '{engine_name}' to real_engines.csv")
        else:
            logging.warning(f"Could not automatically find an .exe for {engine_name}. You may need to add it to real_engines.csv manually.")

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing the file: {e}")


def main():
    """Main function to guide the user through downloading and processing engines."""
    try:
        import py7zr
    except ImportError:
        print("\nThis script requires the 'py7zr' library to handle .7z archives.")
        print("Please install it by running: pip install py7zr\n")
        return

    csv_path = PROJECT_FOLDER / "real_engines.csv"
    if not csv_path.is_file():
        with open(csv_path, "w") as f:
            f.write("path,rating\n")

    print("--- Guided Chess Engine Downloader ---")
    print(f"This script will guide you to download engines.")
    print(f"1. A URL to a 'Releases' page will be provided.")
    print(f"2. Open the URL, find the correct Windows .zip file, and download it.")
    print(f"3. Save the file to the following folder: {DOWNLOADS_DIR}")
    print(f"4. Return to this window and press Enter to continue.")

    for engine in ENGINE_DATABASE:
        print("\n" + "="*60)
        print(f"Engine: {engine['name']} (Rating: {engine['rating']})")
        if engine['notes']:
            print(f"Notes: {engine['notes']}")
        print(f"\nRelease Page URL: {engine['url']}")
        
        input("\n>>> Press Enter after you have downloaded the file to the 'downloads' folder...")
        
        process_downloaded_file(engine)
        
        cont = input("Continue to the next engine? (y/n): ").lower().strip()
        if cont != 'y':
            break
            
    print("\n--- Download process finished. ---")
    print(f"Check the '{ENGINES_DIR}' folder for your new engines.")
    print(f"Review the '{csv_path}' file to ensure the executable paths are correct.")

if __name__ == "__main__":
    main()
