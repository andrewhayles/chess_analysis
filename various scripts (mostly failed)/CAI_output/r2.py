import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# --- User Configuration ---
# 1. Specify the path to your input CSV file.
INPUT_CSV_FILE = 'C:/Users/desja/Documents/Python_programs/chess_study/CAI_output/analysis_results.csv'

# 2. Specify the column containing the full engine name.
#    The script will extract the first number from this column to use as the 'engine rating'.
ENGINE_NAME_COLUMN = 'engine_name'

# 3. Specify the column names for the x and y axes for the analysis.
#    'engine_rating' is the name of the new column that will be created by the script.
X_AXIS_COLUMN = 'engine_rating'  # Independent variable (Engine Rating)
Y_AXIS_COLUMN = 'cai_score'      # Dependent variable (CAI Score)

# 4. Specify the names for the output files.
OUTPUT_ANALYSIS_FILE = 'regression_analysis_results.txt'
OUTPUT_GRAPH_FILE = 'cai_vs_engine_rating_graph.png'
# --- End of Configuration ---


def analyze_relationship(csv_file, name_col, x_col, y_col, analysis_file, graph_file):
    """
    Extracts a numerical rating from a name column, calculates the R-squared value,
    and generates a regression plot.

    Args:
        csv_file (str): The path to the input CSV file.
        name_col (str): The column containing names to extract ratings from.
        x_col (str): The name for the new column holding the extracted rating (x-axis).
        y_col (str): The name of the column for the y-axis (dependent variable).
        analysis_file (str): The path for the output text file with results.
        graph_file (str): The path for the output graph image.
    """
    # --- 1. Load Data ---
    try:
        print(f"Reading data from '{csv_file}'...")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Error: The file '{csv_file}' was not found.")
            
        data = pd.read_csv(csv_file)
        
        # Check if the source name column and y-axis column exist
        if name_col not in data.columns or y_col not in data.columns:
            raise KeyError(f"Error: One or both required columns ('{name_col}', '{y_col}') not found.")

    except FileNotFoundError as e:
        print(e)
        return
    except KeyError as e:
        print(e)
        print(f"Available columns are: {list(data.columns)}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    # --- 2. Extract Engine Rating from Name ---
    print(f"Extracting numerical engine rating from '{name_col}' column...")
    # Use a regular expression to find the first sequence of digits.
    # .str.extract() returns a DataFrame, so we select the first column [0].
    # pd.to_numeric converts the extracted strings to numbers, errors='coerce' turns non-numbers into NaN.
    data[x_col] = pd.to_numeric(data[name_col].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    
    # --- 3. Prepare Data for Regression ---
    # Drop rows where we couldn't extract a rating or where CAI score is missing
    original_rows = len(data)
    data.dropna(subset=[x_col, y_col], inplace=True)
    if len(data) == 0:
        print(f"Error: No valid data rows remaining after attempting to extract ratings from '{name_col}'.")
        print("Please check the contents of the column and the script's logic.")
        return
    print(f"Successfully processed {len(data)} out of {original_rows} rows.")

    # Prepare data for scikit-learn
    # The independent variable (X) needs to be a 2D array
    X = data[[x_col]] 
    y = data[y_col]

    # --- 4. Perform Linear Regression ---
    print("Performing linear regression analysis...")
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # --- 5. Calculate R-squared ---
    r2 = r2_score(y, y_pred)
    print(f"Calculation complete. R-squared value: {r2:.4f}")

    # --- 6. Save Analysis to Text File ---
    print(f"Saving analysis results to '{analysis_file}'...")
    with open(analysis_file, 'w') as f:
        f.write("Linear Regression Analysis Results\n")
        f.write("==================================\n\n")
        f.write(f"Rating extracted from column:  '{name_col}'\n")
        f.write(f"Independent Variable (X-axis): '{x_col}'\n")
        f.write(f"Dependent Variable (Y-axis):   '{y_col}'\n\n")
        f.write(f"R-squared (R²) Value: {r2:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write(f"The R² value indicates that approximately {r2:.2%} of the variance in '{y_col}'\n")
        f.write(f"can be explained by the linear relationship with '{x_col}'.\n")

    # --- 7. Generate and Save Graph ---
    print(f"Generating and saving graph to '{graph_file}'...")
    plt.figure(figsize=(10, 6))
    sns.regplot(x=x_col, y=y_col, data=data, line_kws={"color": "red"})
    
    plt.title(f'Relationship between {y_col.replace("_", " ").title()} and {x_col.replace("_", " ").title()}', fontsize=16)
    plt.xlabel(x_col.replace("_", " ").title(), fontsize=12)
    plt.ylabel(y_col.replace("_", " ").title(), fontsize=12)
    plt.grid(True)
    
    # Add R-squared value to the plot
    plt.text(0.05, 0.95, f'$R^2 = {r2:.4f}$', transform=plt.gca().transAxes,
             fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
             
    plt.savefig(graph_file)
    plt.close() # Close the plot to free up memory

    print("\nAnalysis complete. Check the output files:")
    print(f"- Results: {analysis_file}")
    print(f"- Graph:   {graph_file}")


if __name__ == '__main__':
    # Run the analysis function with the specified configuration
    analyze_relationship(
        csv_file=INPUT_CSV_FILE,
        name_col=ENGINE_NAME_COLUMN,
        x_col=X_AXIS_COLUMN,
        y_col=Y_AXIS_COLUMN,
        analysis_file=OUTPUT_ANALYSIS_FILE,
        graph_file=OUTPUT_GRAPH_FILE
    )
