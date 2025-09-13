import chess
import chess.engine
import chess.pgn
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import multiprocessing
import time

# --- Configuration & Constants ---
ENGINES_CSV_FILE = 'real_engines.csv'
STATE_FILE = 'analysis_state.csv'
ORACLE_CACHE_DIR = 'oracle_cache'
OUTPUT_DIR = 'analysis_output'
LADDER_REPORT_FILE = os.path.join(OUTPUT_DIR, 'ladder_analysis_report.txt')
LADDER_PLOT_FILE = os.path.join(OUTPUT_DIR, 'ladder_analysis_plot.png')

# Statistical significance parameters
DEFAULT_CONFIDENCE = 0.95
DEFAULT_MARGIN_OF_ERROR = 0.015
MIN_POSITIONS = 50
MAX_POSITIONS = 2000

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Helper Functions ---

def z_score_for_confidence(confidence_level):
    alpha = 1 - confidence_level
    return np.abs(np.percentile(np.random.normal(0, 1, 100000), (alpha / 2) * 100))

def calculate_margin_of_error(p, n, z):
    if n == 0 or p < 0 or p > 1: return float('inf')
    p = max(min(p, 0.9999), 0.0001)
    return z * np.sqrt((p * (1 - p)) / n)

def elo_curve(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))


# --- Core Classes ---

class EngineManager:
    def __init__(self, engines_csv_path):
        if not os.path.exists(engines_csv_path):
            raise FileNotFoundError(f"Engine configuration file not found: {engines_csv_path}")
            
        self.engines_df = pd.read_csv(engines_csv_path, comment='#').dropna(subset=['engine_name'])
        self.engines_df['uci_options'] = self.engines_df['uci_options'].apply(json.loads)
        
        oracle_rows = self.engines_df[self.engines_df['engine_name'] == 'stockfish_full']
        if oracle_rows.empty:
            raise ValueError("Configuration Error: Oracle engine 'stockfish_full' not found in real_engines.csv.")
        self.oracle_config = oracle_rows.iloc[0]

        self.test_engines = self.engines_df[self.engines_df['engine_name'] != 'stockfish_full']

    def get_engine(self, config):
        try:
            engine = chess.engine.SimpleEngine.popen_uci(config['path'])
            engine.configure(config['uci_options'])
            return engine
        except Exception as e:
            print(f"\nERROR starting engine '{config['engine_name']}' from path '{config['path']}'.")
            print(f"Error details: {e}")
            return None


class AnalysisState:
    def __init__(self, state_file_path, all_engine_names):
        self.path = state_file_path
        self.lock = multiprocessing.Lock()
        self.columns = ['engine_name', 'positions_analyzed', 'rank0_hits', 'rank1_hits', 'rank2_hits', 'is_complete']
        
        with self.lock:
            if os.path.exists(self.path):
                self.state_df = pd.read_csv(self.path)
            else:
                self.state_df = pd.DataFrame(columns=self.columns)

            new_rows = []
            current_names = set(self.state_df['engine_name'])
            for name in all_engine_names:
                if name not in current_names:
                    new_rows.append({'engine_name': name, 'positions_analyzed': 0, 'rank0_hits': 0, 'rank1_hits': 0, 'rank2_hits': 0, 'is_complete': False})
            
            if new_rows:
                self.state_df = pd.concat([self.state_df, pd.DataFrame(new_rows)], ignore_index=True)
            self.save_nolock()
        
    def get_state(self, engine_name):
        with self.lock:
            return self.state_df[self.state_df['engine_name'] == engine_name].iloc[0].to_dict()

    def update_state(self, engine_name, move_rank):
        with self.lock:
            idx = self.state_df.index[self.state_df['engine_name'] == engine_name].tolist()[0]
            self.state_df.loc[idx, 'positions_analyzed'] += 1
            if move_rank == 0: self.state_df.loc[idx, 'rank0_hits'] += 1
            elif move_rank == 1: self.state_df.loc[idx, 'rank1_hits'] += 1
            elif move_rank == 2: self.state_df.loc[idx, 'rank2_hits'] += 1
            self.save_nolock()
            
    def mark_complete(self, engine_name):
        with self.lock:
            idx = self.state_df.index[self.state_df['engine_name'] == engine_name].tolist()[0]
            self.state_df.loc[idx, 'is_complete'] = True
            self.save_nolock()

    def save_nolock(self):
        self.state_df.to_csv(self.path, index=False)


# --- Main Application Logic ---

class MasterAnalyzer:
    def __init__(self, args):
        self.args = args
        self.z_score = z_score_for_confidence(args.confidence)
        self.preloaded_cache = {}
        
        print("--- Master Chess Analyzer ---")
        os.makedirs(ORACLE_CACHE_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if self.args.import_cache:
            self._load_external_caches()

        self.engine_manager = EngineManager(ENGINES_CSV_FILE)
        
    def _load_external_caches(self):
        print("Importing external oracle caches...")
        for cache_file in self.args.import_cache:
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        count = 0
                        for fen, moves_data in data.items():
                            if isinstance(moves_data, dict) and 'oracle_moves' in moves_data:
                                moves = moves_data['oracle_moves']
                            elif isinstance(moves_data, list):
                                moves = moves_data
                            else: continue
                            self.preloaded_cache[fen] = [{'move': move, 'score': 'N/A'} for move in moves]
                            count += 1
                    print(f"  -> Successfully loaded {count} positions from '{cache_file}'")
                except json.JSONDecodeError:
                    print(f"  -> WARNING: Could not parse JSON from '{cache_file}'.")
                except Exception as e:
                    print(f"  -> WARNING: An error occurred loading '{cache_file}': {e}")
            else:
                print(f"  -> WARNING: Cache file not found: '{cache_file}'")

    def run(self):
        if self.args.mode == 'collect-ladder':
            self.run_ladder_collection()
        elif self.args.mode == 'analyze-ladder':
            self.run_ladder_analysis()
        elif self.args.mode == 'estimate-elo':
            self.run_elo_estimation()
        else:
            print(f"Error: Unknown mode '{self.args.mode}'")

    def _get_oracle_analysis(self, board, oracle_engine, pbar=None):
        fen = board.fen()
        if fen in self.preloaded_cache:
            return self.preloaded_cache[fen]

        fen_slug = fen.replace('/', '_').replace(' ', '_')
        cache_path = os.path.join(ORACLE_CACHE_DIR, f"{fen_slug}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f: return json.load(f)

        if not oracle_engine: return []
        
        if pbar: pbar.set_postfix_str("CACHE MISS: Running oracle... (please wait)")

        try:
            with oracle_engine.analysis(board, chess.engine.Limit(nodes=self.args.oracle_nodes), multipv=3) as analysis:
                result = []
                for info in analysis:
                    if 'pv' in info and info['pv']:
                        result.append({'move': info['pv'][0].uci(), 'score': str(info.get('score', 'N/A'))})
                    if len(result) == 3: break
            with open(cache_path, 'w') as f: json.dump(result, f)
            return result
        except (chess.engine.EngineTerminatedError, BrokenPipeError):
            if pbar: pbar.set_postfix_str("ERROR: Oracle engine failed.")
            print("\nERROR: Oracle engine terminated unexpectedly. Caching failed for this position.")
            return []
            
    def _get_positions_from_pgn(self, pgn_path):
        if not os.path.exists(pgn_path):
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")
        
        positions = []
        try:
            with open(pgn_path) as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None: break
                    board = game.board()
                    for i, move in enumerate(game.mainline_moves()):
                        board.push(move)
                        if (i + 1) >= self.args.skip_moves: positions.append(board.copy())
        except Exception as e:
            print(f"\nWARNING: Could not fully read PGN file '{pgn_path}'. Error: {e}")
        return positions

    def run_ladder_collection(self):
        print("\n--- Starting ELO Ladder Data Collection (Parallel Mode) ---")
        
        opening_positions = self._get_positions_from_pgn(self.args.opening_suite)
        if not opening_positions:
            raise ValueError(f"No usable positions found in the opening suite PGN: '{self.args.opening_suite}'.")

        print(f"Found {len(opening_positions)} unique positions in '{self.args.opening_suite}' for calibration.")
        state = AnalysisState(STATE_FILE, list(self.engine_manager.test_engines['engine_name']))
            
        for _, engine_config in self.engine_manager.test_engines.iterrows():
            engine_state = state.get_state(engine_config['engine_name'])
            if not engine_state['is_complete']:
                self._process_engine_parallel(engine_config.to_dict(), opening_positions, state)
            else:
                print(f"Engine {engine_config['engine_name']} is already complete. Skipping.")
                
        print("\n--- Ladder Data Collection Finished ---")
        
    def _process_engine_parallel(self, config, positions, state):
        engine_name = config['engine_name']
        state_data = state.get_state(engine_name)
        start_position_idx = state_data['positions_analyzed']
        
        positions_to_process = positions[start_position_idx:min(len(positions), MAX_POSITIONS)]
        if not positions_to_process:
            if not state_data['is_complete']: state.mark_complete(engine_name)
            return

        print(f"Processing '{engine_name}'. Analyzing up to {len(positions_to_process)} positions.")
        
        pbar = tqdm(total=len(positions_to_process), desc=f"{engine_name.ljust(20)}")
        
        # Create a pool of workers
        num_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
        num_workers = min(num_workers, 2) # Cap at 2 for this hardware

        with multiprocessing.Pool(processes=num_workers) as pool:
            # Map the worker function to the positions
            results = pool.imap_unordered(self.worker_process, [(pos, config, self.args) for pos in positions_to_process])
            
            for move_rank in results:
                state.update_state(engine_name, move_rank)
                pbar.update(1)
                
                current_state = state.get_state(engine_name)
                n = current_state['positions_analyzed']
                p = current_state['rank0_hits'] / n if n > 0 else 0
                moe = calculate_margin_of_error(p, n, self.z_score)
                
                pbar.set_postfix({"Hit%": f"{p:.2%}", "MoE": f"{moe:.2%}", "Target": f"{self.args.margin_of_error:.2%}"})

                if (n >= MIN_POSITIONS and moe <= self.args.margin_of_error):
                    print(f"\nStatistical significance reached for '{engine_name}' after {n} positions.")
                    pool.terminate() # Stop processing more positions
                    break

        pbar.close()
        state.mark_complete(engine_name)

    @staticmethod
    def worker_process(args_tuple):
        board, test_engine_config, cli_args = args_tuple
        
        # This function runs in a separate process, so it needs its own engine instances
        engine_manager = EngineManager(ENGINES_CSV_FILE)
        oracle_engine = engine_manager.get_engine(engine_manager.oracle_config)
        test_engine = engine_manager.get_engine(test_engine_config)

        move_rank = -1
        try:
            if oracle_engine and test_engine:
                # Re-implement a simplified oracle lookup for the worker
                fen = board.fen()
                fen_slug = fen.replace('/', '_').replace(' ', '_')
                cache_path = os.path.join(ORACLE_CACHE_DIR, f"{fen_slug}.json")

                if os.path.exists(cache_path):
                    with open(cache_path, 'r') as f: oracle_moves_info = json.load(f)
                else:
                    with oracle_engine.analysis(board, chess.engine.Limit(nodes=cli_args.oracle_nodes), multipv=3) as analysis:
                        oracle_moves_info = []
                        for info in analysis:
                            if 'pv' in info and info['pv']:
                                oracle_moves_info.append({'move': info['pv'][0].uci()})
                            if len(oracle_moves_info) == 3: break
                    with open(cache_path, 'w') as f: json.dump(oracle_moves_info, f)

                if len(oracle_moves_info) > 0:
                    oracle_top_moves = [info['move'] for info in oracle_moves_info]
                    move_to_evaluate = test_engine.play(board, chess.engine.Limit()).move.uci()
                    try: move_rank = oracle_top_moves.index(move_to_evaluate)
                    except ValueError: pass
        finally:
            if oracle_engine: oracle_engine.quit()
            if test_engine: test_engine.quit()
        
        return move_rank

    def run_elo_estimation(self):
        print(f"\n--- Estimating ELO from PGN: {self.args.pgn} ---")
        if not os.path.exists(STATE_FILE):
             print("ERROR: Ladder data file (analysis_state.csv) not found. Run 'collect-ladder' mode first.")
             return
             
        player_positions = self._get_positions_from_pgn(self.args.pgn)
        if not player_positions:
            raise ValueError(f"No usable positions found in player PGN: '{self.args.pgn}'.")
        
        oracle_engine = self.engine_manager.get_engine(self.engine_manager.oracle_config)
        player_stats = {'name': os.path.basename(self.args.pgn), 'analyzed': 0, 'r0': 0, 'r1': 0, 'r2': 0}
        
        print("Analyzing player moves against oracle...")
        pbar = tqdm(player_positions, desc="Player PGN")
        try:
            for board in pbar:
                oracle_moves_info = self._get_oracle_analysis(board, oracle_engine, pbar)
                if len(oracle_moves_info) < 1: continue
                oracle_top_moves = [info['move'] for info in oracle_moves_info]
                
                move_to_evaluate = board.peek().uci()
                player_stats['analyzed'] += 1
                try:
                    move_rank = oracle_top_moves.index(move_to_evaluate)
                    if move_rank == 0: player_stats['r0'] += 1
                    elif move_rank == 1: player_stats['r1'] += 1
                    elif move_rank == 2: player_stats['r2'] += 1
                except ValueError: pass
        finally:
            if oracle_engine: oracle_engine.quit()

        if player_stats['analyzed'] == 0:
            print("ERROR: No valid positions found to analyze in the provided PGN file.")
            return

        ladder_data = pd.read_csv(STATE_FILE)
        ladder_data = pd.merge(ladder_data, self.engine_manager.engines_df, on='engine_name', how='left')
        
        weight_scenarios = [
            {'name': 'Top1', 'weights': {'r0': 1.0, 'r1': 0.0, 'r2': 0.0}},
            {'name': 'Balanced', 'weights': {'r0': 1.0, 'r1': 0.5, 'r2': 0.25}},
            {'name': 'Linear', 'weights': {'r0': 1.0, 'r1': 0.66, 'r2': 0.33}},
        ]
        
        print("\n--- ELO Estimation Report ---")
        print(f"PGN File: {player_stats['name']}")
        print(f"Positions Analyzed: {player_stats['analyzed']}")
        print("-" * 30)
        
        all_estimates = []
        for scenario in weight_scenarios:
            name, weights = scenario['name'], scenario['weights']
            
            ladder_data[f'hit_perc'] = ((ladder_data['rank0_hits'] * weights['r0'] + ladder_data['rank1_hits'] * weights['r1'] + ladder_data['rank2_hits'] * weights['r2']) / ladder_data['positions_analyzed']).fillna(0)
            
            player_hit_perc = ((player_stats['r0'] * weights['r0'] + player_stats['r1'] * weights['r1'] + player_stats['r2'] * weights['r2']) / player_stats['analyzed'])
            
            fit_data = ladder_data[ladder_data['positions_analyzed'] > 0].dropna(subset=['rating'])
            est_rating = "N/A"
            if len(fit_data) >= 3:
                try:
                    p0 = [1, 0.01, fit_data['rating'].median()]
                    bounds = ([0, 0, 0], [1.01, 1, 5000])
                    params, _ = curve_fit(elo_curve, fit_data['rating'], fit_data[f'hit_perc'], p0=p0, bounds=bounds, maxfev=10000)
                    a, b, c = params
                    if player_hit_perc > 0 and player_hit_perc < a and (a / player_hit_perc) - 1 > 0:
                        rating = c - (np.log((a / player_hit_perc) - 1) / b)
                        all_estimates.append(rating)
                        est_rating = f"{rating:.0f}"
                except (RuntimeError, ValueError, ZeroDivisionError): pass
            
            print(f"Scenario '{name}': Hit% = {player_hit_perc:.2%}, Estimated ELO = {est_rating}")

        if all_estimates:
            avg_rating = np.mean(all_estimates)
            print("-" * 30)
            print(f"Average Estimated ELO: {avg_rating:.0f}")
            print("-" * 30)

    def run_ladder_analysis(self):
        print("\n--- Analyzing ELO Ladder Performance ---")
        if not os.path.exists(STATE_FILE):
             print("ERROR: Ladder data file (analysis_state.csv) not found. Run 'collect-ladder' mode first.")
             return

        ladder_data = pd.read_csv(STATE_FILE)
        ladder_data = pd.merge(ladder_data, self.engine_manager.engines_df, on='engine_name', how='left')
        
        weight_scenarios = [
            {'name': 'Top1_Only', 'weights': {'r0': 1.0, 'r1': 0.0, 'r2': 0.0}},
            {'name': 'Balanced', 'weights': {'r0': 1.0, 'r1': 0.5, 'r2': 0.25}},
            {'name': 'Linear_Decay', 'weights': {'r0': 1.0, 'r1': 0.66, 'r2': 0.33}},
        ]

        plt.figure(figsize=(14, 8))
        
        with open(LADDER_REPORT_FILE, 'w') as f:
            f.write(f"ELO Ladder Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            for scenario in weight_scenarios:
                name, weights = scenario['name'], scenario['weights']
                
                ladder_data[f'hit_perc'] = ((ladder_data['rank0_hits'] * weights['r0'] + ladder_data['rank1_hits'] * weights['r1'] + ladder_data['rank2_hits'] * weights['r2']) / ladder_data['positions_analyzed']).fillna(0) * 100
                
                f.write(f"--- Scenario: {name} ---\nWeights: {weights}\n\n")
                f.write(ladder_data[['engine_name', 'rating', 'positions_analyzed', 'hit_perc']].rename(columns={'hit_perc': 'Hit %'}).round(2).to_string(index=False))
                f.write("\n\n" + "-"*80 + "\n\n")

                fit_data = ladder_data[ladder_data['positions_analyzed'] > 0].dropna(subset=['rating'])
                params = None
                if len(fit_data) >= 3:
                    try:
                        p0 = [100, 0.01, fit_data['rating'].median()]
                        bounds = ([0, 0, 0], [101, 1, 5000])
                        params, _ = curve_fit(elo_curve, fit_data['rating'], fit_data[f'hit_perc'], p0=p0, bounds=bounds, maxfev=10000)
                    except RuntimeError: params = None
                
                plt.scatter(fit_data['rating'], fit_data[f'hit_perc'], label=f'{name} - Engine Data', alpha=0.8)

                if params is not None:
                    x_fit = np.linspace(min(fit_data['rating']), max(fit_data['rating']), 300)
                    y_fit = elo_curve(x_fit, *params)
                    plt.plot(x_fit, y_fit, linestyle='--', label=f'{name} - Fitted Curve')
        
        plt.title('Engine Performance vs. Calibrated Elo Rating', fontsize=16)
        plt.xlabel('Engine Elo Rating', fontsize=12)
        plt.ylabel('Weighted "Oracle Move" Hit Percentage (%)', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        plt.savefig(LADDER_PLOT_FILE)
        print(f"Successfully saved plot to '{LADDER_PLOT_FILE}'")
        print(f"Analysis report saved to '{LADDER_REPORT_FILE}'")


def main():
    parser = argparse.ArgumentParser(description="A comprehensive, resumable chess engine analysis tool.", formatter_class=argparse.RawTextHelpFormatter)
    
    subparsers = parser.add_subparsers(dest='mode', required=True, help='The mode to run')

    parser_collect = subparsers.add_parser('collect-ladder', help='Builds the statistical model by collecting data for the ELO ladder engines.')
    parser_collect.add_argument("--opening-suite", default="openings.pgn", help="Path to a PGN with diverse positions for engine testing. (Default: openings.pgn)")
    
    parser_analyze = subparsers.add_parser('analyze-ladder', help='Generates a report and plot visualizing the performance of the ELO ladder.')

    parser_estimate = subparsers.add_parser('estimate-elo', help='Estimates the ELO of a player based on a PGN file of their games.')
    parser_estimate.add_argument("--pgn", required=True, help="Path to the PGN file containing the player's games to be analyzed.")

    for p in [parser_collect, parser_analyze, parser_estimate]:
        p.add_argument("--oracle-nodes", type=int, default=5_000_000, help="Nodes for the oracle engine to search per position. (Default: 5,000,000)")
        p.add_argument("--skip-moves", type=int, default=16, help="Half-moves (plies) to skip from the start of each game. (Default: 16)")
        p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help=f"Confidence level for statistical significance. (Default: {DEFAULT_CONFIDENCE})")
        p.add_argument("--margin-of-error", type=float, default=DEFAULT_MARGIN_OF_ERROR, help=f"Target margin of error to stop data collection. (Default: {DEFAULT_MARGIN_OF_ERROR})")
        p.add_argument("--import-cache", type=str, action='append', help="Path to an existing oracle cache JSON file to import. Can be used multiple times.")

    args = parser.parse_args()
    
    try:
        analyzer = MasterAnalyzer(args)
        analyzer.run()
    except (ValueError, FileNotFoundError) as e:
        print(f"\nFATAL ERROR: {e}")
        print("Script cannot continue. Please correct the configuration and try again.")


if __name__ == "__main__":
    multiprocessing.freeze_support() # Recommended for PyInstaller/Windows
    main()

