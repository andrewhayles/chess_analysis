import json
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import os

def parse_engine_rating(engine_name):
    """
    Extracts the Elo rating from an engine name string.
    It looks for names containing '_elo_' or 'maia_' followed by a number.
    Returns the rating as an integer or None if not found.
    """
    # Regex to find numbers in strings like 'maia_1100' or 'stockfish_elo_2007'
    match = re.search(r'(?:_elo_|maia_|_)([1-9][0-9]{2,3})$', engine_name)
    if match:
        return int(match.group(1))
    return None

def calculate_hit_rates(model_moves, ground_truth, template_size):
    """
    Calculates the hit rates for each engine based on a given template size.
    A 'hit' is when an engine's move is within the top N moves of the ground truth.

    Args:
        model_moves (dict): Dictionary of moves predicted by engines for each position.
        ground_truth (dict): Dictionary of top moves for each position.
        template_size (int): The number of top moves to consider (e.g., 1, 2, 3...).

    Returns:
        list: A list of dictionaries, each containing an engine's name, rating, and hit rate.
    """
    engine_stats = {}

    # Initialize stats for all engines that have a parsable rating
    if not model_moves:
        return []
        
    first_pos_key = next(iter(model_moves))
    for engine_name in model_moves[first_pos_key]:
        rating = parse_engine_rating(engine_name)
        if rating is not None:
            engine_stats[engine_name] = {'hits': 0, 'total': 0, 'rating': rating}

    # Iterate over all game positions (FENs)
    for fen, engine_predictions in model_moves.items():
        if fen not in ground_truth:
            continue

        # Get the list of top N moves from the ground truth template
        top_n_moves = [move_info['move'] for move_info in ground_truth[fen][:template_size]]

        # Check each engine's prediction
        for engine_name, predicted_move in engine_predictions.items():
            if engine_name in engine_stats:
                engine_stats[engine_name]['total'] += 1
                if predicted_move in top_n_moves:
                    engine_stats[engine_name]['hits'] += 1

    # Calculate the final hit rate for each engine
    results = []
    for engine_name, stats in engine_stats.items():
        if stats['total'] > 0:
            hit_rate = stats['hits'] / stats['total']
            results.append({
                'engine': engine_name,
                'rating': stats['rating'],
                'hit_rate': hit_rate
            })
            
    # Sort results by rating for cleaner plotting
    return sorted(results, key=lambda x: x['rating'])


def create_plot(data, method_name, template_size, output_dir):
    """
    Creates and saves a scatter plot of Hit Rate vs. Rating.

    Args:
        data (list): The processed data from calculate_hit_rates.
        method_name (str): The name of the method (e.g., 'depth_first').
        template_size (int): The template size used for the calculation.
        output_dir (str): The directory to save the plot image.
    """
    if not data:
        print(f"No data to plot for {method_name}, template size {template_size}.")
        return

    ratings = [d['rating'] for d in data]
    hit_rates = [d['hit_rate'] for d in data]
    engine_names = [d['engine'] for d in data]

    # Calculate the Pearson correlation coefficient
    # Requires at least two data points
    if len(ratings) > 1:
        corr, _ = pearsonr(ratings, hit_rates)
        corr_text = f"Correlation: {corr:.4f}"
    else:
        corr_text = "Correlation: N/A"


    plt.figure(figsize=(12, 8))
    plt.scatter(ratings, hit_rates, s=120, alpha=0.7, edgecolors='w', label='Engines')
    
    # Add labels to each point on the scatter plot
    for i, name in enumerate(engine_names):
        # Slightly offset the text to avoid overlapping with the point
        plt.text(ratings[i], hit_rates[i] + 0.01, name, fontsize=9, ha='center')

    # Set plot titles and labels
    plt.title(f'Hit Rate vs. Rating ({method_name.replace("_", " ").title()})\nTemplate Size: {template_size}, {corr_text}')
    plt.xlabel('Engine Rating (Elo)')
    plt.ylabel(f'Hit Rate (in Top {template_size} Moves)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(-0.05, 1.05) # Give some padding to the y-axis
    
    # Add a linear regression trend line if there are enough points
    if len(ratings) > 1:
        z = np.polyfit(ratings, hit_rates, 1)
        p = np.poly1d(z)
        # Generate a smooth line for the trend
        plot_x = np.linspace(min(ratings), max(ratings), 100)
        plt.plot(plot_x, p(plot_x), "r--", label=f'Trendline')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot to a file
    filename = f'{method_name}_template_{template_size}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Saved plot to {filepath}")
    plt.close()


def main():
    """
    Main function to load data, run the analysis, and generate plots.
    """
    json_file = 'goalgames.session.json'
    output_dir = 'analysis_graphs'

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the JSON data from the file
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file}' was not found.")
        print("Please make sure the script is in the same directory as the JSON file.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file}'. Please check the file format.")
        return

    # Extract the necessary data structures from the JSON
    game_data = data.get('in_progress_game', {})
    ground_truth = game_data.get('ground_truth_template')
    depth_moves = game_data.get('model_engine_moves_depth_first')
    time_moves = game_data.get('model_engine_moves_time_first')

    if not all([ground_truth, depth_moves, time_moves]):
        print("Error: The JSON file is missing required data sections ('ground_truth_template', 'model_engine_moves_depth_first', or 'model_engine_moves_time_first').")
        return

    methods = {
        'depth_first': depth_moves,
        'time_first': time_moves
    }
    template_sizes = [1, 2, 3, 4, 5]

    # Loop through each method and template size to generate a plot
    for method_name, model_moves in methods.items():
        for size in template_sizes:
            print("-" * 30)
            print(f"Processing: {method_name}, Template Size: {size}")
            hit_rate_data = calculate_hit_rates(model_moves, ground_truth, size)
            create_plot(hit_rate_data, method_name, size, output_dir)
    
    print("-" * 30)
    print(f"Analysis complete. All graphs have been saved to the '{output_dir}' directory.")

if __name__ == '__main__':
    main()
