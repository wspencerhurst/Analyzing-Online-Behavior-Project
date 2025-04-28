import os
import json
import pandas as pd
from dateutil import parser as date_parser
from dateutil import tz
from datetime import datetime, timedelta
import re
from collections import Counter
#import torch
#from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import google.generativeai as genai
import time

# --- Configuration ---
# --- START OF CHANGES ---
# Input CSV file path relative to the script location
INPUT_CSV_PATH = os.path.join("data", "google-history", "jackson", "final-takeout-data.csv")
OUTPUT_CSV_PATH = "activity_summary.csv"
# --- END OF CHANGES ---

# Local timezone - lmk if you went anywhere crazy
local_tz = tz.gettz("America/New_York")

# Threshold (minutes) for inactivity gap to consider as sleep
SLEEP_INACTIVITY_THRESHOLD = 300  # 5 hours

# Words to ignore in activity content
STOPWORDS = {"watched", "watch", "google", "youtube", "search", "com", "www", "https", "http"}

# --- Gemini API Configuration ---
# !!! IMPORTANT: Replace "<API_KEY>" with your actual Google API key !!!
# --- START OF CHANGES ---
# Check if API key is set as an environment variable first
genai.configure(api_key="AIzaSyB9X1LSI6qsL2HpjVt_njEBNdTol30Wg3Y")
model = genai.GenerativeModel('gemini-1.5-flash')

# --- END OF CHANGES ---


# --- Original LLM Functions (Kept as is, but added checks for model availability) ---
def summarize_titles(titles, max_tokens=100):
    # --- START OF CHANGES ---
    if not model:
        print("Skipping summarization: Gemini model not available.")
        return "Summarization disabled"
    # --- END OF CHANGES ---
    try:
        activities = " ".join(titles)
        prompt = (
            "Summarize the following activities into exactly 3 to 5 short bullet points. "
            "Each bullet point should describe a broad activity type based on the following categories: studying, work, health and fitness, entertainment, or relaxation. "
            "Do NOT copy video titles. Only use 2 to 6 words per bullet point.\n\n"
            f"Activities:\n{activities}\n\n"
            "Summary bullet points:"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Warning: Summarization failed. Error: {e}")
        return "Summarization Error" # Return specific error string

def rate_stress_level(titles, max_tokens=10):
    # --- START OF CHANGES ---
    if not model:
        print("Skipping stress rating: Gemini model not available.")
        return -1 # Return default value
    # --- END OF CHANGES ---
    try:
        activities = " ".join(titles)
        prompt = (
            "Estimate the percentage of activities that involve academic studying, work tasks, coding projects, finance work, or job-related activities. "
            "If 70% or more of activities are work-related, rate stress between 7–10. "
            "If between 30% and 70% are work-related, rate stress between 4–6. "
            "If less than 30% are work-related, rate stress between 1–3. "
            "Respond ONLY with a single number from 1 to 10.\n\n"
            f"Activities:\n{activities}\n\n"
            "Stress rating:"
        )
        response = model.generate_content(prompt)
        text = response.text.strip()
        match = re.search(r'\b([1-9]|10)\b', text)
        if match:
            return int(match.group(0))
        else:
            # Handle cases where the model doesn't return a number correctly
            print(f"Warning: Could not parse stress rating from response: '{text}'")
            return -1
    except Exception as e:
        print(f"Warning: Stress rating failed. Error: {e}")
        return -1

# --- Original Helper Function (Kept as is) ---
def extract_phrases(ngrams, titles, stopwords, top_n=10):
    text = " ".join(titles).lower()
    # Only keep words of at least 4 letters
    words = re.findall(r'\b\w{4,}\b', text)
    words = [w for w in words if w not in stopwords]
    if ngrams == 2:
        grams = zip(words, words[1:])
    elif ngrams == 3: # Corrected logic for 3-grams
        grams = zip(words, words[1:], words[2:])
    else: # Handle cases other than 2 or 3 ngrams
        return ""
    phrases = [" ".join(gram) for gram in grams]
    counter = Counter(phrases)
    return ", ".join([phrase for phrase, count in counter.most_common(top_n)])


# --- Main Processing Logic ---
def main():
    # --- START OF CHANGES ---
    # Read data directly from the specified CSV file
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input CSV file not found at {INPUT_CSV_PATH}")
        return

    print(f"Reading data from {INPUT_CSV_PATH}...")
    try:
        raw_df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Successfully read {len(raw_df)} rows.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Filter for relevant activity types (mimicking original script's sources)
    # We map 'Search' (from CSV header) to 'Chrome' for consistency with original output.
    # We keep 'YouTube' as 'YouTube'.
    print("Filtering data for 'Search' and 'YouTube' headers...")
    relevant_headers = ['Search', 'YouTube']
    filtered_df = raw_df[raw_df['header'].isin(relevant_headers)].copy()
    print(f"Filtered down to {len(filtered_df)} relevant rows.")

    if filtered_df.empty:
        print("No relevant 'Search' or 'YouTube' activities found in the CSV.")
        return

    # Prepare the DataFrame to match the structure expected by the original logic
    all_usage = pd.DataFrame()

    # Convert time strings to datetime objects and handle potential errors
    try:
        # Assuming 'time' column is in a format pandas can recognize (like ISO 8601)
        # Convert to UTC first if not already, then to local timezone
        all_usage['datetime'] = pd.to_datetime(filtered_df['time'], errors='coerce').dt.tz_convert(local_tz)
    except Exception as e:
         print(f"Error converting 'time' column to datetime: {e}. Trying dateutil parser...")
         # Fallback using dateutil parser if standard pandas fails
         datetimes = []
         for t_str in filtered_df['time']:
             try:
                 # Parse assumes UTC if no timezone info, then convert
                 dt_utc = date_parser.parse(t_str)
                 if dt_utc.tzinfo is None:
                     dt_utc = dt_utc.replace(tzinfo=tz.UTC)
                 datetimes.append(dt_utc.astimezone(local_tz))
             except Exception as parse_err:
                 print(f"Could not parse time string: {t_str}. Error: {parse_err}. Skipping row.")
                 datetimes.append(pd.NaT) # Add NaT for failed parses
         all_usage['datetime'] = datetimes


    # Map 'header' to 'source' ('Search' -> 'Chrome', 'YouTube' -> 'YouTube')
    source_mapping = {'Search': 'Chrome', 'YouTube': 'YouTube'}
    all_usage['source'] = filtered_df['header'].map(source_mapping)

    # Assign 'title' and 'url' (renaming 'titleUrl')
    all_usage['title'] = filtered_df['title'].fillna('') # Fill NaN titles
    all_usage['url'] = filtered_df['titleUrl'].fillna('')   # Fill NaN URLs

    # Select and reorder columns to match the original structure precisely
    all_usage = all_usage[['datetime', 'source', 'title', 'url']]

    # Drop rows where datetime conversion failed
    initial_rows = len(all_usage)
    all_usage = all_usage.dropna(subset=["datetime"])
    dropped_rows = initial_rows - len(all_usage)
    if dropped_rows > 0:
        print(f"Warning: Dropped {dropped_rows} rows due to invalid timestamps after parsing.")

    # --- END OF CHANGES ---

    # --- Original Logic (mostly unchanged from here) ---
    if all_usage.empty:
        print("No valid activity data found after processing the CSV.")
        return

    print("Sorting activities by datetime...")
    all_usage = all_usage.sort_values("datetime").reset_index(drop=True)

    # --- Date Range Cleaning (Original logic kept) ---
    # This step might be less critical if the input CSV is already clean,
    # but keeping it ensures compatibility if unexpected dates exist.
    before_cleaning = len(all_usage)
    all_usage = all_usage[
        (all_usage["datetime"] >= datetime(2000, 1, 1, tzinfo=local_tz)) &
        (all_usage["datetime"] <= datetime(2100, 1, 1, tzinfo=local_tz))
    ]
    after_cleaning = len(all_usage)
    if before_cleaning != after_cleaning:
        print(f"Warning: Dropped {before_cleaning - after_cleaning} rows with timestamps outside 2000-2100 range.")

    if all_usage.empty:
        print("No activity found within the valid date range (2000-2100).")
        return

    print("Calculating time differences and identifying sleep gaps...")
    # Calculate time difference between consecutive activities
    all_usage["time_diff_min"] = all_usage["datetime"].diff().dt.total_seconds() / 60

    # Identify large gaps, potentially indicating sleep
    # The first gap is always NaN, so we rely on gaps *after* the first activity
    sleep_gaps = all_usage[all_usage["time_diff_min"] > SLEEP_INACTIVITY_THRESHOLD].reset_index()

    if sleep_gaps.empty:
        print("No sleep gaps identified based on the threshold. Cannot process day-by-day.")
        # Decide if you want to process the entire dataset as one "day" or exit
        # For compatibility, we'll exit if no gaps are found.
        return
    else:
         print(f"Identified {len(sleep_gaps)} potential sleep gaps.")


    results = {} # Dictionary to store results per day

    print(f"Source distribution in filtered data:\n{all_usage['source'].value_counts()}")

    # Iterate through sleep gaps to define 'days' (from wake-up to next bedtime)
    # The loop goes up to len(sleep_gaps) because the last gap defines the end of the *previous* day.
    print("Processing days based on sleep gaps...")
    for i in tqdm(range(len(sleep_gaps)), desc="Processing Days"):
        # The start time of the day is the datetime *after* the previous sleep gap
        # For the very first day (i=0), the start time is the first activity in the dataset
        start_index = 0 if i == 0 else sleep_gaps.loc[i-1, "index"] + 1 # Index in all_usage AFTER the gap ends
        if start_index >= len(all_usage): # Should not happen if sleep_gaps isn't empty, but safe check
            continue
        prev_wake = all_usage.loc[start_index, "datetime"]

        # The end time of the day is the datetime *of* the current sleep gap start
        # This activity is the last one *before* the sleep gap begins.
        end_index = sleep_gaps.loc[i, "index"]
        curr_bed = all_usage.loc[end_index, "datetime"]

        # Define the day based on the *start* of the sleep period (curr_bed)
        # A "day" represents activity leading up to that sleep.
        day = curr_bed.date() # Use the date of the bedtime as the key for the day's activity

        # --- DATE FILTERING: Only process days within the specified range ---
        # The original script seemed to filter based on the *end* of the activity period (curr_bed)
        target_start_date = datetime(2025, 1, 20, tzinfo=local_tz)
        target_end_date   = datetime(2025, 4, 20, 23, 59, 59, tzinfo=local_tz) # Inclusive end date

        if not (target_start_date <= curr_bed <= target_end_date):
            #print(f"Skipping day {day}: Bedtime {curr_bed} outside target range.")
            continue
        # --- END DATE FILTERING ---


        # Filter activities for this specific day period
        # Use index slicing for efficiency since it's sorted
        day_activities = all_usage.loc[start_index:end_index]

        if day_activities.empty:
            #print(f"No activities found for day {day} (Period: {prev_wake} to {curr_bed}). Skipping.")
            continue

        # Calculate metrics for the day
        earliest_dt = day_activities["datetime"].min()
        latest_dt   = day_activities["datetime"].max()
        # Convert times to minutes since midnight for the output columns
        earliest    = earliest_dt.hour * 60 + earliest_dt.minute
        latest      = latest_dt.hour   * 60 + latest_dt.minute
        total_count = len(day_activities)
        # Use the mapped 'source' column
        youtube_count = (day_activities["source"] == "YouTube").sum()
        chrome_count = (day_activities["source"] == "Chrome").sum()

        # Prepare titles for analysis
        titles = day_activities["title"].dropna().astype(str).tolist()

        # Generate summaries and ratings
        summary_keywords = extract_phrases(1, titles, STOPWORDS) # Use 1 for single words
        summary_bigrams = extract_phrases(2, titles, STOPWORDS)
        summary_trigrams = extract_phrases(3, titles, STOPWORDS)

        # --- LLM Calls with Rate Limiting ---
        # Added checks here again for safety, although checked at the start
        if model:
            try:
                summary = summarize_titles(titles)
                time.sleep(4)  # Wait 4 seconds before the next API call
                stress_rating = rate_stress_level(titles)
                time.sleep(4)  # Wait 4 seconds after the API call
            except Exception as llm_err:
                 print(f"Error during LLM calls for day {day}: {llm_err}")
                 summary = "LLM Error"
                 stress_rating = -1
        else:
            summary = "LLM Disabled"
            stress_rating = -1
        # --- End LLM Calls ---


        # Store results for the day
        # The original script's logic for overwriting if a day appears multiple
        # times seems flawed if based on total_count. Let's just store the first
        # encounter for a given date, assuming the sleep gap logic correctly defines days.
        # If a day appears multiple times, it might indicate issues with the gap detection or data.
        # For now, let's keep it simple and just assign/overwrite.
        results[day] = {
            "earliest_activity_time": earliest,
            "latest_activity_time": latest,
            "total_activity_count": total_count,
            "youtube_activity_count": youtube_count,
            "chrome_activity_count": chrome_count,
            "keywords": summary_keywords,
            "keybigrams": summary_bigrams,
            "keytrigrams": summary_trigrams,
            "summary": summary,
            "llm_stress_rating": stress_rating
        }
        # print(f"Processed day {day}: Start={prev_wake}, End={curr_bed}, Activities={total_count}")


    # Convert results dictionary to DataFrame and save
    if not results:
        print("No results generated within the target date range (Jan 20 - Apr 20, 2025). No output file created.")
        return

    print(f"Aggregated results for {len(results)} days.")
    df_out = pd.DataFrame([
        {"day": str(day), **vals} for day, vals in sorted(results.items())
    ])

    # Ensure columns match the original output exactly
    expected_columns = [
        "day",
        "earliest_activity_time",
        "latest_activity_time",
        "total_activity_count",
        "youtube_activity_count",
        "chrome_activity_count",
        "keywords",
        "keybigrams",
        "keytrigrams",
        "summary",
        "llm_stress_rating"
    ]
    df_out = df_out[expected_columns] # Reorder/select columns to match

    print(f"Saving results to {OUTPUT_CSV_PATH}...")
    try:
        df_out.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        print(f"Successfully saved activity summary to {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")

if __name__ == "__main__":
    main()