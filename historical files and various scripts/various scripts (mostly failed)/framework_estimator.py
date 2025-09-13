# framework_visualizer.py
#
# This script reads the completed analysis from 'optimizer_session.json' and
# generates detailed performance graphs for each scoring framework (CAI, RWPL, etc.).
#
# --- HOW IT WORKS ---
# 1. Loads the session data, engine data, and constants.
# 2. Calculates the average score for each engine under every scoring framework.
# 3. For each framework, it builds a 3rd-degree polynomial regression model to
#    correlate engine scores with their Elo ratings.
# 4. It then generates and saves a scatter plot for each framework, showing:
#    - The individual engine performances (dots).
#    - The fitted regression curve (line).
#    - The R-squared value, indicating the model's predictive power.
#    - The full equation of the regression curve.
#
# --- USAGE ---
# 1. Place this script in the same directory as your 'final_optimizer.py' and
#    the 'chess_analysis_session' folder.
# 2. Run from the terminal: python framework_visualizer.py
# 3. The graphs will be saved in a new 'framework_graphs' subfolder.

import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import csv
from collections import defaultdict
import logging

# --- CONFIGURATION ---
PROJECT_FOLDER = Path(__file__).resolve().parent
SESSION_FOLDER = PROJECT_FOLDER / "chess_analysis_session"
GRAPHS_FOLDER = PROJECT_FOLDER / "framework_graphs" # Folder to save the plots
SESSION_FILE = SESSION_FOLDER / "optimizer_session.json"
ENGINES_CSV_PATH = PROJECT_FOLDER / "real_engines.csv"
CONSTANTS_FILE = SESSION_FOLDER / "optimized_constants.json"

# --- FRAMEWORK SETTINGS (Should match the optimizer script) ---
HITCOUNT_MAX_TEMPLATE_SIZE = 5

# --- SCRIPT SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- HELPER & CALCULATION FUNCTIONS (from final_optimizer.py) ---

def load_json_file(file_path, default_data=None):
    """Loads a JSON file with error handling."""
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not read or parse {file_path}: {e}. Using default data.")
            return default_data or {}
    return default_data or {}

DEFAULT_CONSTANTS = {
    "K1_WPL": 0.00368, "K2_RWPL": 0.5, "K3_CAI": 0.004, "W_CAI": 1.0,
    "HITCOUNT_SECOND_MOVE_WEIGHT": 0.75
}

def calculate_move_scores_from_data(move_data, constants):
    """Calculates scores for a single move across all frameworks using pre-calculated data."""
    scores = {}
    
    best_move_eval = move_data['best_eval']
    played_move_eval = move_data['played_eval']
    if played_move_eval is not None and best_move_eval is not None:
        second_best_move_eval = move_data['second_best_eval']
        wp_best = 1 / (1 + np.exp(-constants['K1_WPL'] * best_move_eval))
        wp_played = 1 / (1 + np.exp(-constants['K1_WPL'] * played_move_eval))
        wp_loss = max(0, wp_best - wp_played)
        scores['RWPL'] = 100 * (1 - 2 * (min(wp_loss, 0.5) ** constants['K2_RWPL']))
        eval_gap = abs(best_move_eval - second_best_move_eval) if second_best_move_eval is not None else 0
        criticality = 1 / (1 + np.exp(-constants['K3_CAI'] * eval_gap))
        impact = wp_loss * (1 + constants['W_CAI'] * criticality)
        scores['CAI'] = 100 * (1 - 2 * (min(impact, 0.5) ** constants['K2_RWPL']))
        scores['Impact'] = scores['CAI'] # Impact is an alias for CAI in the optimizer script

    engine_specific_evals = move_data.get('complexity_evals')
    scores['Complexity'] = np.std(engine_specific_evals) if engine_specific_evals and len(engine_specific_evals) > 1 else 0.0

    played_move = move_data.get('played_move')
    oracle_moves = move_data.get('oracle_moves', [])
    if played_move and oracle_moves:
        for t_size in range(1, HITCOUNT_MAX_TEMPLATE_SIZE + 1):
            framework_name = f"HitCount_T{t_size}"
            score = 0.0
            template_slice = oracle_moves[:t_size]
            if played_move in template_slice:
                move_index = template_slice.index(played_move)
                if move_index == 0: score = 1.0
                elif move_index == 1: score = constants['HITCOUNT_SECOND_MOVE_WEIGHT']
            scores[framework_name] = score
            
    return scores

def get_all_simulation_data(session_data):
    """Processes the raw session data into a list of move data points for engines."""
    processed_data = []
    all_sim_results = session_data.get('all_sim_results', {})
    all_ground_truth = session_data.get('all_ground_truth', {})
    for fen, template in all_ground_truth.items():
        if not template or fen not in all_sim_results: continue
        oracle_moves = [m['move'] for m in template]
        for engine_name, move_data in all_sim_results[fen].items():
            processed_data.append({
                'engine_name': engine_name,
                'best_eval': template[0]['eval'],
                'second_best_eval': template[1]['eval'] if len(template) > 1 else None,
                'played_eval': move_data.get('eval'), 'played_move': move_data.get('move'),
                'oracle_moves': oracle_moves, 'complexity_evals': move_data.get('complexity_evals')
            })
    return processed_data

def get_model_equation(model_pipeline, degree=3):
    """Extracts the coefficients and formats the polynomial equation string."""
    lr = model_pipeline.named_steps['linearregression']
    poly_features = model_pipeline.named_steps['polynomialfeatures']
    
    # The coefficients from the linear model
    coeffs = lr.coef_
    intercept = lr.intercept_
    
    # The powers of the polynomial features (e.g., [0], [1], [2], [3] for x^0, x^1, x^2, x^3)
    powers = poly_features.powers_

    equation = f"y = {intercept:.4f}"
    for i in range(1, len(coeffs)):
        power = powers[i][0]
        coeff = coeffs[i]
        if coeff >= 0:
            equation += f" + {coeff:.4e}x^{power}"
        else:
            equation += f" - {abs(coeff):.4e}x^{power}"
            
    return equation

# --- MAIN VISUALIZATION LOGIC ---

def main():
    logging.info("--- Framework Performance Visualizer ---")

    # 1. Load data
    if not SESSION_FILE.exists():
        logging.error(f"FATAL: Session file not found at {SESSION_FILE}. Run the optimizer first."); return
    if not ENGINES_CSV_PATH.exists():
        logging.error(f"FATAL: Engines CSV not found at {ENGINES_CSV_PATH}."); return

    session_data = load_json_file(SESSION_FILE)
    constants = load_json_file(CONSTANTS_FILE, DEFAULT_CONSTANTS)
    all_engines = {row['name']: int(row['rating']) for row in csv.DictReader(open(ENGINES_CSV_PATH)) if not row['name'].strip().startswith('#')}
    
    GRAPHS_FOLDER.mkdir(exist_ok=True)
    logging.info(f"Loaded data for {len(session_data.get('processed_game_ids', []))} games. Graphs will be saved to '{GRAPHS_FOLDER}'.")

    # 2. Process data and calculate scores for all frameworks
    all_processed_data = get_all_simulation_data(session_data)
    engine_scores_by_framework = defaultdict(lambda: defaultdict(list))

    for move_data in all_processed_data:
        move_scores = calculate_move_scores_from_data(move_data, constants)
        for name, score in move_scores.items():
            engine_scores_by_framework[name][move_data['engine_name']].append(score)
    
    avg_scores_by_framework = {fw: {eng: np.mean(sc) for eng, sc in es.items() if sc} for fw, es in engine_scores_by_framework.items()}

    # 3. Generate a plot for each framework
    for framework_name, scores_dict in avg_scores_by_framework.items():
        logging.info(f"Generating graph for framework: '{framework_name}'...")
        
        if not scores_dict:
            logging.warning(f"  -> No scores found for '{framework_name}'. Skipping.")
            continue

        eng_data = [(all_engines[name], score) for name, score in scores_dict.items() if name in all_engines]
        if len(eng_data) < 3:
            logging.warning(f"  -> Not enough data points ({len(eng_data)}) for '{framework_name}'. Skipping.")
            continue

        ratings_data, scores_data = zip(*eng_data)
        ratings = np.array(ratings_data).reshape(-1, 1)
        scores = np.array(scores_data)

        # 4. Build model and get R² and equation
        model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        model.fit(ratings, scores)
        r_squared = r2_score(scores, model.predict(ratings))
        equation = get_model_equation(model)

        # 5. Create and save the plot
        plt.figure(figsize=(12, 8))
        plt.scatter(ratings, scores, alpha=0.7, label="Engine Performance")
        
        plot_x = np.linspace(ratings.min(), ratings.max(), 200).reshape(-1, 1)
        plt.plot(plot_x, model.predict(plot_x), color='red', lw=2, label=f"Polynomial Fit (R² = {r_squared:.4f})")
        
        plt.title(f"Performance of '{framework_name}' Framework vs. Engine Rating", fontsize=16)
        plt.xlabel("Engine Elo Rating", fontsize=12)
        plt.ylabel(f"Average '{framework_name}' Score", fontsize=12)
        plt.grid(True)
        
        # Add the equation text to the plot
        plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        
        plt.legend()
        
        graph_path = GRAPHS_FOLDER / f"framework_performance_{framework_name}.png"
        plt.savefig(graph_path)
        plt.close() # Close the figure to free up memory
        logging.info(f"  -> Saved graph to {graph_path}")

    logging.info("--- All graphs generated successfully. ---")

if __name__ == "__main__":
    main()
