# engine_downloader.py
import requests
import zipfile
import py7zr
from pathlib import Path
import io
import logging

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(r"C:\Users\desja\Documents\Python_programs\chess_study")
ENGINES_DIR = PROJECT_FOLDER / "engines"

# --- ENGINE DOWNLOAD LIST ---
# Add engines to this dictionary. Use direct download links where possible.
# Key: A simple name for the engine
# Value: The direct URL to the .zip or .7z file
ENGINES_TO_DOWNLOAD = {
    "stockfish_17.1": "https://stockfishchess.org/files/stockfish-17.1-win-x86-64-avx2.zip",
    "komodo_14.1": "https://komodochess.com/pub/komodo-14.1-64bit.zip",
    # Example for a hypothetical engine:
    # "ExampleEngine": "http://some-website.com/engines/ExampleEngine_v2.7z"
}

# --- SCRIPT SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ENGINES_DIR.mkdir(exist_ok=True)

def download_and_extract(engine_name, url):
    """Downloads an engine archive and extracts it."""
    try:
        logging.info(f"Downloading {engine_name} from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Will raise an exception for bad status codes

        content_type = response.headers.get('content-type', '')
        
        # Use io.BytesIO to treat the downloaded content as a file in memory
        file_bytes = io.BytesIO(response.content)
        
        extract_path = ENGINES_DIR / engine_name
        extract_path.mkdir(exist_ok=True)
        
        logging.info(f"Extracting to {extract_path}...")

        if 'zip' in content_type or url.endswith('.zip'):
            with zipfile.ZipFile(file_bytes, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif 'x-7z-compressed' in content_type or url.endswith('.7z'):
            with py7zr.SevenZipFile(file_bytes, mode='r') as z_ref:
                z_ref.extractall(path=extract_path)
        else:
            logging.warning(f"Unsupported archive type for {engine_name}. Manual extraction needed.")
            return

        logging.info(f"Successfully downloaded and extracted {engine_name}.")
        
        # Attempt to find the .exe file
        executables = list(extract_path.glob('**/*.exe'))
        if executables:
            logging.info(f"Found executable(s): {[exe.name for exe in executables]}")
        else:
            logging.warning(f"Could not automatically find an .exe for {engine_name}.")

    except requests.RequestException as e:
        logging.error(f"Failed to download {engine_name}. Error: {e}")
    except (zipfile.BadZipFile, py7zr.Bad7zFile) as e:
        logging.error(f"Failed to extract {engine_name}. Archive may be corrupt. Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred for {engine_name}: {e}")

def main():
    """Main function to download all specified engines."""
    logging.info("Starting engine download process...")
    
    # You will need py7zr for .7z files: pip install py7zr
    print("This script requires the 'py7zr' library for .7z files.")
    print("If not installed, please run: pip install py7zr")
    
    for name, url in ENGINES_TO_DOWNLOAD.items():
        download_and_extract(name, url)
        
    logging.info("Download process finished.")
    logging.info(f"Please check the '{ENGINES_DIR}' folder.")
    logging.info("You may need to manually move executables out of subfolders.")

if __name__ == "__main__":
    main()