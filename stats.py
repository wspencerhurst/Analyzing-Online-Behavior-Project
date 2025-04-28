import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

BASE_DATA_PATH = './'
PROCESSED_DATA_DIR = BASE_DATA_PATH
PEOPLE = ['Spencer', 'Joseph', 'Jackson']

warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(person_name, data_dir):
    file_path = os.path.join(data_dir, f"{person_name.lower()}_processed.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"Successfully loaded data for {person_name} from {file_path}")
            return df
        except Exception as e:
            print(f"Error reading file for {person_name} at {file_path}: {e}")
            return None
    else:
        print(f"Data file not found for {person_name} at {file_path}")
        return None

def analyze_person_data(df, name):
    if df is None or df.empty:
        print(f"\n--- No data to analyze for {name} ---")
        return

    print(f"\n--- Analyzing Data for: {name} ---")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    target_cols = ['stress', 'llm_stress_rating']
    valid_target_cols = [col for col in target_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    cols_for_corr = list(set(numeric_cols + valid_target_cols))

    if not cols_for_corr:
        print(f"No numeric columns found for correlation analysis for {name}.")
        return

    numeric_df = df[cols_for_corr].copy()
    initial_rows = len(numeric_df)
    print(f"Initial numeric rows for correlation: {initial_rows}")
    cols_to_keep = [col for col in numeric_df.columns if numeric_df[col].fillna(0).abs().sum() > 0]
    if len(cols_to_keep) < len(numeric_df.columns):
        dropped_cols = set(numeric_df.columns) - set(cols_to_keep)
        print(f"Dropping columns with only zero/NaN values: {', '.join(dropped_cols)}")
        numeric_df = numeric_df[cols_to_keep]
    else:
        print("No columns found containing only zero/NaN values.")

    if numeric_df.empty or numeric_df.shape[1] < 2:
        print(f"Not enough columns with non-zero data remaining for correlation analysis for {name}.")
        return
    is_all_zero = (numeric_df == 0).all(axis=1)
    numeric_df = numeric_df[~is_all_zero]
    filtered_rows = len(numeric_df)

    is_all_zero = (numeric_df == 0).all(axis=1)
    numeric_df = numeric_df[~is_all_zero]
    filtered_rows = len(numeric_df)
    print(f"Rows after filtering out all-zero rows: {filtered_rows} ({initial_rows - filtered_rows} rows removed)")

    if numeric_df.empty or filtered_rows < 2:
        print(f"Not enough non-zero data remaining for correlation analysis for {name}.")
        return

    print(f"\nCalculating Correlation Matrix for {name}...")
    numeric_df.dropna(axis=1, how='all', inplace=True)
    if numeric_df.empty or numeric_df.shape[1] < 2:
         print(f"Not enough numeric data remaining for correlation analysis for {name} after dropping all-NaN columns.")
         return

    corr_matrix = numeric_df.corr()

    print(f"Generating Heatmap for {name}...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5, annot_kws={"size": 8})
    plt.title(f'Correlation Matrix Heatmap - {name} (All-Zero Rows Excluded)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    print(f"\nCorrelations with Stress Metrics for {name} (All-Zero Rows Excluded):")

    if 'stress' in corr_matrix.columns:
        stress_corr = corr_matrix['stress'].sort_values(ascending=False)
        print("\nCorrelations with 'stress':")
        print(stress_corr)
    else:
        print("\n'stress' column not found or not numeric for correlation analysis.")

    if 'llm_stress_rating' in corr_matrix.columns:
        llm_stress_corr = corr_matrix['llm_stress_rating'].sort_values(ascending=False)
        print("\nCorrelations with 'llm_stress_rating':")
        print(llm_stress_corr)
    else:
        print("\n'llm_stress_rating' column not found or not numeric for correlation analysis.")

    print(f"\nDescriptive Statistics for Numeric Columns ({name}, All-Zero Rows Excluded):")
    try:
        print(numeric_df.describe())
    except Exception as e:
        print(f"Could not generate descriptive statistics: {e}")


if __name__ == "__main__":
    all_data = {}
    loaded_people = []

    for person in PEOPLE:
        df_person = load_data(person, PROCESSED_DATA_DIR)
        if df_person is not None:
            all_data[person] = df_person
            loaded_people.append(person)

    if not all_data:
        print("\nNo data files found in the processed directory. Exiting.")
    else:
        print("\n\n====== INDIVIDUAL ANALYSIS ======")
        for person_name, df in all_data.items():
            analyze_person_data(df, person_name)

        print("\n\n====== COMBINED ANALYSIS (All Loaded People) ======")
        if len(loaded_people) > 0:
            combined_df = pd.concat(all_data.values(), ignore_index=False, sort=False)
            print(f"Combined DataFrame shape: {combined_df.shape}")

            numeric_combined_df = combined_df.select_dtypes(include=np.number).copy()

            max_missing_threshold = 0.50 # Keep columns with at least 50% non-NaN data
            cols_before_nan_filter = numeric_combined_df.shape[1]
            nan_threshold = len(numeric_combined_df) * (1 - max_missing_threshold)
            numeric_combined_df.dropna(axis=1, thresh=nan_threshold, inplace=True)
            cols_after_nan_filter = numeric_combined_df.shape[1]
            if cols_after_nan_filter < cols_before_nan_filter:
                print(f"Filtered columns in combined data due to >{max_missing_threshold*100}% missing values.")
                print(f"Columns remaining: {numeric_combined_df.columns.tolist()}")

            analyze_person_data(numeric_combined_df, "Combined (Filtered Columns)")

        else:
            print("No data loaded, skipping combined analysis.")

        print("\nDisplaying generated heatmaps...")
        plt.show()