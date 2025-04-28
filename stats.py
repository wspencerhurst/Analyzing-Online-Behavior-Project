import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
from wordcloud import WordCloud, STOPWORDS

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
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5, annot_kws={"size": 8}, vmin=-1, vmax=1)
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


def generate_stress_wordclouds(person_name, data_dir, output_dir):
    """Generates and saves/shows word clouds for keywords based on LLM stress rating."""
    print(f"\n--- Generating Word Clouds for {person_name} ---")
# Construct the path directly
    history_csv_path = os.path.join(BASE_DATA_PATH, 'data', 'google-history', person_name.lower(), f"{person_name.lower()}_activity_summary.csv")
    if not os.path.exists(history_csv_path):
        print(f"History file not found for {person_name} at {history_csv_path}. Skipping word clouds.")
        return

    try:
        stopwords = set(STOPWORDS)
        custom_stopwords = ["searched"]
        all_stopwords = stopwords.union(custom_stopwords)

        # Load the specific activity summary CSV
        df_hist = pd.read_csv(history_csv_path)

        # --- Data Preparation ---
        # Check for stress rating first
        if 'llm_stress_rating' not in df_hist.columns:
            print(f"Required column 'llm_stress_rating' not found in {history_csv_path}. Skipping word clouds.")
            return

        # Determine which keyword column to use
        keyword_col_name = 'keywords' # Default
        keywords_exist_and_has_data = ('keywords' in df_hist.columns and df_hist['keywords'].notna().any())

        if person_name == 'Jackson' and not keywords_exist_and_has_data:
            print(f"Keywords column missing or empty for Jackson. Trying 'keybigrams'.")
            if 'keybigrams' in df_hist.columns and df_hist['keybigrams'].notna().any():
                keyword_col_name = 'keybigrams' # Switch to keybigrams for Jackson
                print(f"Using 'keybigrams' column for Jackson.")
            else:
                print(f"Neither 'keywords' nor 'keybigrams' found or has data for Jackson in {history_csv_path}. Skipping word clouds.")
                return
        elif not keywords_exist_and_has_data: # For people other than Jackson, keywords must exist and have data
            print(f"Required column 'keywords' not found or empty in {history_csv_path}. Skipping word clouds for {person_name}.")
            return
        # else: Use the default 'keywords'

        # Convert stress rating to numeric, drop rows where it's NaN
        df_hist['llm_stress_rating'] = pd.to_numeric(df_hist['llm_stress_rating'], errors='coerce')
        # Drop rows missing stress OR the chosen keyword column
        df_hist.dropna(subset=['llm_stress_rating', keyword_col_name], inplace=True)

        # Fill NaN in the chosen keyword column with empty string AFTER dropping rows
        df_hist[keyword_col_name] = df_hist[keyword_col_name].fillna('')

        # Define stress thresholds
        high_stress_threshold = 5

        # --- Separate Keywords by Stress Level ---
        # Use the determined keyword column name
        high_stress_keywords = df_hist[df_hist['llm_stress_rating'] > high_stress_threshold][keyword_col_name]
        low_stress_keywords = df_hist[df_hist['llm_stress_rating'] <= high_stress_threshold][keyword_col_name]

        # Combine keywords into single text blobs, replacing commas with spaces
        # Note: This assumes keybigrams are also comma-separated like keywords were
        high_stress_text = ' '.join(high_stress_keywords).replace(',', ' ').lower()
        low_stress_text = ' '.join(low_stress_keywords).replace(',', ' ').lower()

        # --- Generate Word Clouds ---
        # Create figure for side-by-side plots
        fig_wc, axes_wc = plt.subplots(1, 2, figsize=(16, 8)) # 1 row, 2 columns

        # High Stress Word Cloud
        if high_stress_text.strip(): # Check if there's any text after cleaning
            print(f"Generating high stress word cloud for {person_name}...")
            wc_high = WordCloud(stopwords=all_stopwords, width=800, height=600, background_color='white', colormap='Reds').generate(high_stress_text)
            axes_wc[0].imshow(wc_high, interpolation='bilinear')
            axes_wc[0].set_title(f'{person_name} - High Stress Keywords (> {high_stress_threshold})')
            axes_wc[0].axis('off')
        else:
            print(f"No keywords found for high stress days for {person_name}.")
            axes_wc[0].text(0.5, 0.5, 'No High Stress Keywords Found', horizontalalignment='center', verticalalignment='center', transform=axes_wc[0].transAxes)
            axes_wc[0].set_title(f'{person_name} - High Stress Keywords (> {high_stress_threshold})')
            axes_wc[0].axis('off')


        # Low Stress Word Cloud
        if low_stress_text.strip(): # Check if there's any text after cleaning
            print(f"Generating low stress word cloud for {person_name}...")
            wc_low = WordCloud(stopwords=all_stopwords, width=800, height=600, background_color='white', colormap='Blues').generate(low_stress_text)
            axes_wc[1].imshow(wc_low, interpolation='bilinear')
            axes_wc[1].set_title(f'{person_name} - Low/Moderate Stress Keywords (<= {high_stress_threshold})')
            axes_wc[1].axis('off')
        else:
             print(f"No keywords found for low/moderate stress days for {person_name}.")
             axes_wc[1].text(0.5, 0.5, 'No Low/Mod Stress Keywords Found', horizontalalignment='center', verticalalignment='center', transform=axes_wc[1].transAxes)
             axes_wc[1].set_title(f'{person_name} - Low/Moderate Stress Keywords (<= {high_stress_threshold})')
             axes_wc[1].axis('off')

        plt.tight_layout()

        # --- Save the Word Cloud Figure ---
        try:
            # Ensure output directory exists (should have been created earlier)
            os.makedirs(output_dir, exist_ok=True)
            wc_fig_path = os.path.join(output_dir, f'{person_name.lower()}_stress_wordclouds.png')
            fig_wc.savefig(wc_fig_path, dpi=150, bbox_inches='tight')
            print(f"Word cloud plot saved to: {wc_fig_path}")
        except Exception as e:
            print(f"Error saving word cloud plot for {person_name}: {e}")

    except FileNotFoundError:
        print(f"Error: History file not found for {person_name} at {history_csv_path} (checked again in word cloud function).")
    except Exception as e:
        print(f"An unexpected error occurred generating word clouds for {person_name}: {e}")


# --- Comparative Time Series Plotting Function ---
def plot_comparative_time_series(all_data, loaded_people, output_dir):
    """Generates comparative time series plots for key metrics across individuals."""
    print("\n--- Generating Comparative Time Series Plots ---")

    if not loaded_people:
        print("No loaded people to generate time series for.")
        return

    # Define the variables for each subplot in the two figures
    fig1_vars = [
        ['stress', 'llm_stress_rating'], # Top row: Stress metrics
        ['restingHeartRate', 'hrv']      # Bottom row: HR metrics
    ]
    fig2_vars = [
        ['stress', 'llm_stress_rating'], # Top row: Stress metrics
        ['event_count', 'academic_count']# Bottom row: Activity counts
    ]

    # --- Create Figure 1 ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10), sharex=True) # Share X-axis (time)
    fig1.suptitle('Comparative Time Series: Stress & Physiological Metrics', fontsize=16)
    axes1 = axes1.flatten() # Flatten for easier indexing

    plot_index = 0
    for row_vars in fig1_vars:
        for var in row_vars:
            ax = axes1[plot_index]
            data_plotted = False
            for person in loaded_people:
                df = all_data.get(person)
                if df is not None and var in df.columns and df[var].notna().any():
                    #(ax=ax, label=person, alpha=0.8) # Use alpha for slight transparency
                    # Calculate and plot the 5-day rolling average
                    df[var].rolling(window=5, center=True, min_periods=1).mean().plot(ax=ax, label=person, alpha=0.8)
                    data_plotted = True

            if data_plotted:
                ax.set_title(f'{var} Over Time')
                ax.set_ylabel(var)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6) # Add subtle grid
            else:
                # Handle case where no person has data for this var
                ax.text(0.5, 0.5, f'No data for {var}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f'{var} Over Time')
            ax.tick_params(axis='x', rotation=45) # Rotate date labels if needed
            plot_index += 1

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

    # --- Save Figure 1 ---
    try:
        fig1_path = os.path.join(output_dir, 'timeseries_comparison_stress_physio.png')
        fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
        print(f"Time series plot 1 saved to: {fig1_path}")
    except Exception as e:
        print(f"Error saving time series plot 1: {e}")


    # --- Create Figure 2 ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10), sharex=True) # Share X-axis (time)
    fig2.suptitle('Comparative Time Series: Stress & Activity Metrics', fontsize=16)
    axes2 = axes2.flatten() # Flatten for easier indexing

    plot_index = 0
    for row_vars in fig2_vars:
        for var in row_vars:
            ax = axes2[plot_index]
            data_plotted = False
            for person in loaded_people:
                df = all_data.get(person)
                if df is not None and var in df.columns and df[var].notna().any():
                    #df[var].plot(ax=ax, label=person, alpha=0.8)
                    # Calculate and plot the 5-day rolling average
                    df[var].rolling(window=5, center=True, min_periods=1).mean().plot(ax=ax, label=person, alpha=0.8)
                    data_plotted = True

            if data_plotted:
                ax.set_title(f'{var} Over Time')
                ax.set_ylabel(var)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
            else:
                # Handle case where no person has data for this var
                ax.text(0.5, 0.5, f'No data for {var}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f'{var} Over Time')
            ax.tick_params(axis='x', rotation=45)
            plot_index += 1

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

    # --- Save Figure 2 ---
    try:
        fig2_path = os.path.join(output_dir, 'timeseries_comparison_stress_activity.png')
        fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
        print(f"Time series plot 2 saved to: {fig2_path}")
    except Exception as e:
        print(f"Error saving time series plot 2: {e}")



# --- Stress vs LLM Stress Rating Comparison Function ---
def plot_stress_comparison(all_data, combined_df, loaded_people, output_dir):
    """Generates 2x2 scatter plots comparing 'stress' and 'llm_stress_rating'."""
    print("\n--- Generating Stress vs LLM Stress Rating Comparison Plot ---")

    # We need at least one person with both columns
    can_plot_individual = False
    for person in loaded_people:
        df = all_data.get(person)
        if df is not None and 'stress' in df.columns and 'llm_stress_rating' in df.columns:
             # Check if there's overlap after dropping NaNs for this pair
             if not df[['stress', 'llm_stress_rating']].dropna().empty:
                 can_plot_individual = True
                 break # Found at least one person

    # Check combined data
    can_plot_combined = False
    if combined_df is not None and 'stress' in combined_df.columns and 'llm_stress_rating' in combined_df.columns:
        if not combined_df[['stress', 'llm_stress_rating']].dropna().empty:
            can_plot_combined = True

    if not can_plot_individual and not can_plot_combined:
        print("Insufficient data: Cannot generate stress comparison plots (need 'stress' and 'llm_stress_rating' columns with overlapping data).")
        return

    # Create Figure (2x2 grid)
    fig_sc, axes_sc = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True) # Share axes for comparison
    axes_sc = axes_sc.flatten()
    fig_sc.suptitle('Stress (Device) vs. LLM Stress Rating Comparison', fontsize=16)

    plot_idx = 0
    # Plot Individuals
    for person in loaded_people:
        if plot_idx >= 3: break # Only plot the first 3 people individually if more are loaded

        ax = axes_sc[plot_idx]
        df = all_data.get(person)

        if df is not None and 'stress' in df.columns and 'llm_stress_rating' in df.columns \
           and pd.api.types.is_numeric_dtype(df['stress']) \
           and pd.api.types.is_numeric_dtype(df['llm_stress_rating']):
             # Check for sufficient overlapping data points
             plot_data = df[['stress', 'llm_stress_rating']].dropna()
             if len(plot_data) >= 2: # Need at least 2 points for regression
                 sns.regplot(x='stress', y='llm_stress_rating', data=plot_data, ax=ax,
                             scatter_kws={'alpha': 0.6, 's': 20}, # Adjust point size and transparency
                             line_kws={'color': 'red', 'lw': 2}) # Adjust line color and width
                 ax.set_title(f'{person}')
                 ax.grid(True, linestyle='--', alpha=0.6)
             else:
                 ax.text(0.5, 0.5, 'Insufficient Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                 ax.set_title(f'{person}')
                 ax.axis('off') # Hide axis if no data
        else:
            # Handle missing columns or non-numeric types
            ax.text(0.5, 0.5, 'Data Missing', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'{person}')
            ax.axis('off')
        plot_idx += 1

    # Plot Combined Data in the last subplot (index 3)
    ax = axes_sc[3]
    if combined_df is not None and 'stress' in combined_df.columns and 'llm_stress_rating' in combined_df.columns \
       and pd.api.types.is_numeric_dtype(combined_df['stress']) \
       and pd.api.types.is_numeric_dtype(combined_df['llm_stress_rating']):
         # Check for sufficient overlapping data points in combined set
         plot_data_combined = combined_df[['stress', 'llm_stress_rating']].dropna()
         if len(plot_data_combined) >= 2:
             sns.regplot(x='stress', y='llm_stress_rating', data=plot_data_combined, ax=ax,
                         scatter_kws={'alpha': 0.4, 's': 15}, # Make combined points slightly different
                         line_kws={'color': 'blue', 'lw': 2})
             ax.set_title('Combined')
             ax.grid(True, linestyle='--', alpha=0.6)
         else:
             ax.text(0.5, 0.5, 'Insufficient Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             ax.set_title('Combined')
             ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'Data Missing', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('Combined')
        ax.axis('off')

    # Hide unused axes if fewer than 3 people loaded
    for i in range(len(loaded_people), 3):
         fig_sc.delaxes(axes_sc[i])

    # Add shared axis labels (since sharex=True, sharey=True)
    fig_sc.text(0.5, 0.02, 'Stress (Device)', ha='center', va='center', fontsize=12)
    fig_sc.text(0.02, 0.5, 'LLM Stress Rating', ha='center', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95]) # Adjust layout for suptitle and shared labels

    # --- Save the Stress Comparison Figure ---
    try:
        sc_fig_path = os.path.join(output_dir, 'stress_comparison_scatter.png')
        fig_sc.savefig(sc_fig_path, dpi=150, bbox_inches='tight')
        print(f"Stress comparison plot saved to: {sc_fig_path}")
    except Exception as e:
        print(f"Error saving stress comparison plot: {e}")

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
            generate_stress_wordclouds(person_name, PROCESSED_DATA_DIR, PROCESSED_DATA_DIR)
            plot_comparative_time_series(all_data, loaded_people, PROCESSED_DATA_DIR)


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

            plot_stress_comparison(all_data, combined_df, loaded_people, PROCESSED_DATA_DIR)

            combined_df.reset_index(inplace=True)
            if 'index' in combined_df.columns:
                combined_df.rename(columns={'index': 'date'}, inplace=True) 
            elif 'calendarDate' in combined_df.columns and combined_df['calendarDate'].duplicated().any():
                pass

            print(f"Combined DataFrame shape after resetting index: {combined_df.shape}")
            print("Columns after resetting index:", combined_df.columns)

            print("\n--- Generating Distribution Comparison Plots ---")

            distribution_vars = [
                'stress', 'llm_stress_rating', 'hrv', 'restingHeartRate',
                'total_duration', 'event_count', 'exercise_duration_weighted',
                'total_activity_count', 'youtube_activity_count', 'chrome_activity_count',
                'early_late_duration', 'academic_count', 'assignment_count',
                'assignment_difficulty_sum'
            ]

            plot_df = combined_df[combined_df['person'].isin(loaded_people)].copy()

            n_vars = len(distribution_vars)
            n_cols = 3
            n_rows = (n_vars + n_cols - 1) // n_cols

            fig_dist, axes_dist = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False) # Use squeeze=False for consistent indexing
            axes_dist = axes_dist.flatten()

            plot_count = 0
            for i, var in enumerate(distribution_vars):
                if var in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[var]):
                    if plot_df[var].notna().sum() > 0:
                        ax = axes_dist[plot_count]
                        sns.boxplot(x='person', y=var, data=plot_df, ax=ax, order=loaded_people)
                        ax.set_title(f'Distribution of {var}')
                        ax.set_xlabel('Person')
                        ax.set_ylabel(var)
                        ax.tick_params(axis='x', rotation=45)
                        plot_count += 1
                    else:
                        print(f"Skipping boxplot for '{var}' due to all NaN values.")
                else:
                    print(f"Skipping boxplot for '{var}' as it's not numeric or not found in combined data.")
            
            for j in range(plot_count, len(axes_dist)):
                fig_dist.delaxes(axes_dist[j])
            plt.suptitle('Distribution Comparisons Across Individuals', fontsize=16, y=1.02)
            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            dist_fig_path = os.path.join(BASE_DATA_PATH, 'distribution_comparison.png')
            fig_dist.savefig(dist_fig_path, dpi=300, bbox_inches='tight')

        else:
            print("No data loaded, skipping combined analysis.")

        print("\nDisplaying generated heatmaps...")
        #plt.show()