import pandas as pd
import json
import os
import ast
from datetime import datetime, time, timedelta
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
print(f"Script Directory: {SCRIPT_DIR}")
BASE_DATA_PATH = SCRIPT_DIR # Assume 'data' is relative 

# Spencer
SPENCER_HR_PATH = os.path.join(BASE_DATA_PATH, 'data', 'smartwatch', 'spencer', 'Heart Rate.json')
SPENCER_HRV_PATH = os.path.join(BASE_DATA_PATH, 'data', 'smartwatch', 'spencer', 'HRV.json')
SPENCER_STRESS_PATH = os.path.join(BASE_DATA_PATH, 'data', 'smartwatch', 'spencer', 'Stress.json')
SPENCER_TIMELINE_PATH = os.path.join(BASE_DATA_PATH, 'data', 'timelines', 'spencer_timeline.json')
# Joseph
JOSEPH_HEALTH_PATH = os.path.join(BASE_DATA_PATH, 'data', 'joseph_processed_oura_data_proj3.csv')
JOSEPH_TIMELINE_PATH = os.path.join(BASE_DATA_PATH, 'data', 'joseph_clean_google_calendar_proj3.csv')
# Jackson
JACKSON_DATA_PATH = os.path.join(BASE_DATA_PATH, 'data', 'jackson-data.csv') # Combined file

def get_history_path(person_name):
    person_lower = person_name.lower()
    filename = f"{person_lower}_activity_summary.csv"
    return os.path.join(BASE_DATA_PATH, 'data', 'google-history', person_lower, filename)

def add_google_history_features(df, person_name):
    print(f"\n--- Adding Google History Features for {person_name} ---")
    history_csv_path = get_history_path(person_name)
    history_cols = ['total_activity_count', 'youtube_activity_count',
                    'chrome_activity_count', 'llm_stress_rating']

    if os.path.exists(history_csv_path):
        try:
            history_df = pd.read_csv(history_csv_path)
            history_df['day'] = pd.to_datetime(history_df['day'])
            history_df.set_index('day', inplace=True)

            history_df_selected = pd.DataFrame(index=history_df.index)
            for col in history_cols:
                 if col in history_df.columns:
                     history_df_selected[col] = pd.to_numeric(history_df[col], errors='coerce')
                 else:
                     print(f"Warning: Column '{col}' not found in {history_csv_path} for {person_name}. Filling with NaN.")
                     history_df_selected[col] = np.nan

            df = pd.merge(df, history_df_selected, left_index=True, right_index=True, how='left')
            print(f"Successfully merged history features for {person_name}.")

        except FileNotFoundError:
            print(f"Warning: History file not found for {person_name} at {history_csv_path}. Adding NaN columns.")
            for col in history_cols:
                df[col] = np.nan
        except Exception as e:
            print(f"Error processing history file for {person_name} at {history_csv_path}: {e}")
            print("Adding NaN columns for history features.")
            for col in history_cols:
                df[col] = np.nan
    else:
        print(f"Warning: History file not found for {person_name} at {history_csv_path}. Adding NaN columns.")
        for col in history_cols:
            df[col] = np.nan
    return df


def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError, TypeError):
            return []
    elif pd.isna(val):
         return []
    return val

def parse_time_range(time_str, date_obj):
    start_dt, end_dt = pd.NaT, pd.NaT
    if not isinstance(time_str, str) or '-' not in time_str:
        return start_dt, end_dt

    try:
        start_str, end_str = time_str.split('-')
        start_time_obj = pd.to_datetime(start_str.strip(), format='%I:%M%p').time()
        end_time_obj = pd.to_datetime(end_str.strip(), format='%I:%M%p').time()

        start_dt = datetime.combine(date_obj.date(), start_time_obj)
        end_dt = datetime.combine(date_obj.date(), end_time_obj)

        # Handle overnight events
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
        return start_dt, end_dt
    except ValueError:
        return pd.NaT, pd.NaT


def calculate_duration_from_dt(start_dt, end_dt):
    if pd.isna(start_dt) or pd.isna(end_dt):
        return 0
    return (end_dt - start_dt).total_seconds() / 60

def process_spencer_data():
    print("\n--- Processing Spencer's Data ---")
    spencer_health_data = {} # {date_str: {metric: value}}

    print("Processing Spencer's Health Data...")
    try:
        with open(SPENCER_HR_PATH, 'r') as f:
            hr_data = json.load(f)
        for entry in hr_data:
            date_str = entry.get('calendarDate')
            if date_str:
                resting_hr = entry.get('values', {}).get('restingHR')
                if date_str not in spencer_health_data: spencer_health_data[date_str] = {}
                spencer_health_data[date_str]['restingHeartRate'] = resting_hr
    except FileNotFoundError:
        print(f"Warning: Spencer HR file not found at {SPENCER_HR_PATH}")
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {SPENCER_HR_PATH}")

    try:
        with open(SPENCER_HRV_PATH, 'r') as f:
            hrv_data = json.load(f)
        for entry in hrv_data.get('hrvSummaries', []):
             date_str = entry.get('calendarDate')
             if date_str:
                 hrv_val = entry.get('lastNightAvg')
                 if date_str not in spencer_health_data: spencer_health_data[date_str] = {}
                 spencer_health_data[date_str]['hrv'] = hrv_val
    except FileNotFoundError:
        print(f"Warning: Spencer HRV file not found at {SPENCER_HRV_PATH}")
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {SPENCER_HRV_PATH}")

    try:
        with open(SPENCER_STRESS_PATH, 'r') as f:
            stress_data = json.load(f)
        for entry in stress_data:
            date_str = entry.get('calendarDate')
            if date_str:
                stress_val = entry.get('values', {}).get('overallStressLevel')
                if date_str not in spencer_health_data: spencer_health_data[date_str] = {}
                spencer_health_data[date_str]['stress'] = stress_val if stress_val is not None and stress_val >= 0 else np.nan
    except FileNotFoundError:
        print(f"Warning: Spencer Stress file not found at {SPENCER_STRESS_PATH}")
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {SPENCER_STRESS_PATH}")

    health_df = pd.DataFrame.from_dict(spencer_health_data, orient='index')
    health_df.index = pd.to_datetime(health_df.index)
    health_df = health_df.sort_index()
    print("Spencer Health Data Preview:")
    print(health_df.head())
    print(health_df.info())

    print("\nProcessing Spencer's Timeline Data...")
    spencer_timeline_features = {}
    try:
        with open(SPENCER_TIMELINE_PATH, 'r') as f:
            timeline_data = json.load(f)

        for day_entry in timeline_data:
            date_str = day_entry.get('date')
            if not date_str: continue
            date_obj = pd.to_datetime(date_str)

            events = day_entry.get('events', {})
            academic_events = events.get('academic', [])
            work_events = events.get('work', [])
            exercise_events = events.get('exercise', [])
            activity_events = events.get('activities', [])

            event_count = 0
            total_duration = 0
            academic_count = 0
            assignment_count = 0
            assignment_difficulty_sum = 0
            exercise_duration_weighted = 0
            early_late_duration = 0 # Duration of events starting before 9am or ending after 9pm

            for event in academic_events:
                event_count += 1
                academic_count += 1
                duration = 0
                start_dt, end_dt = pd.NaT, pd.NaT
                if 'start_time' in event and 'end_time' in event:
                    try:
                        start_time_obj = pd.to_datetime(event['start_time'], format='%H:%M').time()
                        end_time_obj = pd.to_datetime(event['end_time'], format='%H:%M').time()
                        start_dt = datetime.combine(date_obj.date(), start_time_obj)
                        end_dt = datetime.combine(date_obj.date(), end_time_obj)
                        if end_dt < start_dt: end_dt += timedelta(days=1) # Handle overnight
                        duration = calculate_duration_from_dt(start_dt, end_dt)
                    except ValueError:
                        start_dt, end_dt = parse_time_range(f"{event['start_time']}-{event['end_time']}", date_obj)
                        duration = calculate_duration_from_dt(start_dt, end_dt)

                total_duration += duration
                if not pd.isna(start_dt) and (start_dt.time() < time(9, 0) or (not pd.isna(end_dt) and end_dt.time() > time(21, 0) and end_dt.date() == start_dt.date()) or (not pd.isna(end_dt) and end_dt.time() <= time(9,0) and end_dt.date() > start_dt.date())): # check end time > 9pm on same day or ends early next day
                    early_late_duration += duration

                if event.get('type') == 'assignment_due':
                    assignment_count += 1
                    assignment_difficulty_sum += event.get('difficulty', 0)

            for event in work_events:
                event_count += 1
                duration = event.get('duration_minutes', 0)
                total_duration += duration

            for event in exercise_events:
                event_count += 1
                duration = event.get('duration_minutes', 0)
                exertion = event.get('exertion', 1)
                total_duration += duration
                exercise_duration_weighted += duration * exertion

            for event in activity_events:
                event_count += 1
                duration = event.get('duration_minutes', 0)
                total_duration += duration

            spencer_timeline_features[date_obj] = {
                'event_count': event_count,
                'total_duration': total_duration,
                'academic_count': academic_count,
                'assignment_count': assignment_count,
                'assignment_difficulty_sum': assignment_difficulty_sum,
                'exercise_duration_weighted': exercise_duration_weighted,
                'early_late_duration': early_late_duration
            }

        timeline_df = pd.DataFrame.from_dict(spencer_timeline_features, orient='index')
        timeline_df.index = pd.to_datetime(timeline_df.index)
        timeline_df = timeline_df.sort_index()
        print("\nSpencer Timeline Features Preview:")
        print(timeline_df.head())
        print(timeline_df.info())

        spencer_df = pd.merge(health_df, timeline_df, left_index=True, right_index=True, how='outer')
        spencer_df['person'] = 'Spencer'
        print("\nCombined Spencer DataFrame Preview:")
        print(spencer_df.head())
        print(spencer_df.info())
        return spencer_df

    except FileNotFoundError:
        print(f"Error: Spencer timeline file not found at {SPENCER_TIMELINE_PATH}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {SPENCER_TIMELINE_PATH}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred processing Spencer's timeline: {e}")
        return pd.DataFrame()



def process_joseph_data():
    print("\n--- Processing Joseph's Data ---")

    print("Processing Joseph's Health Data...")
    try:
        health_df = pd.read_csv(JOSEPH_HEALTH_PATH)
        health_df['date'] = pd.to_datetime(health_df['date'])
        health_df.rename(columns={
            'Average Resting Heart Rate': 'restingHeartRate',
            'Average HRV': 'hrv',
            'Stress Score': 'stress'
        }, inplace=True)
        health_df = health_df[['date', 'person', 'restingHeartRate', 'hrv', 'stress']].copy()
        health_df.set_index('date', inplace=True)
        health_df = health_df.sort_index()
        print("Joseph Health Data Preview:")
        print(health_df.head())
        print(health_df.info())
    except FileNotFoundError:
        print(f"Error: Joseph health file not found at {JOSEPH_HEALTH_PATH}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred processing Joseph's health data: {e}")
        return pd.DataFrame()

    print("\nProcessing Joseph's Timeline Data...")
    joseph_timeline_features = {} # {date_obj: {feature: value}}
    try:
        timeline_raw_df = pd.read_csv(JOSEPH_TIMELINE_PATH)
        timeline_raw_df['date'] = pd.to_datetime(timeline_raw_df['date'])

        for date_obj, group in timeline_raw_df.groupby('date'):
            event_count = len(group)
            total_duration = group['duration'].sum()
            academic_count = group[group['event_type'] == 'academics'].shape[0]
            # Joseph's timeline doesn't explicitly mark assignments due or difficulty
            assignment_count = academic_count
            assignment_difficulty_sum = 0

            exercise_duration_weighted = group[group['event_type'] == 'exercise']['duration'].sum() # No exertion data

            early_late_duration = 0
            for _, row in group.iterrows():
                start_str = f"{date_obj.date()} {row['start_time']}"
                end_str = f"{date_obj.date()} {row['end_time']}"
                start_dt = pd.to_datetime(start_str, format='%Y-%m-%d %H:%M', errors='coerce')
                end_dt = pd.to_datetime(end_str, format='%Y-%m-%d %H:%M', errors='coerce')

                if pd.isna(start_dt) or pd.isna(end_dt):
                    continue

                if end_dt < start_dt :
                     if (start_dt.time() > time(12,0)) and (end_dt.time() < time(12,0)):
                          end_dt += timedelta(days=1)

                duration = row['duration']

                if (start_dt.time() < time(9, 0)) or \
                   (end_dt.time() > time(21, 0) and end_dt.date() == start_dt.date()) or \
                   (end_dt.time() <= time(9,0) and end_dt.date() > start_dt.date()): # handle overnight ending early
                    early_late_duration += duration


            joseph_timeline_features[date_obj] = {
                'event_count': event_count,
                'total_duration': total_duration,
                'academic_count': academic_count,
                'assignment_count': assignment_count,
                'assignment_difficulty_sum': assignment_difficulty_sum,
                'exercise_duration_weighted': exercise_duration_weighted,
                'early_late_duration': early_late_duration
            }

        timeline_df = pd.DataFrame.from_dict(joseph_timeline_features, orient='index')
        timeline_df.index = pd.to_datetime(timeline_df.index)
        timeline_df = timeline_df.sort_index()
        print("\nJoseph Timeline Features Preview:")
        print(timeline_df.head())
        print(timeline_df.info())


        joseph_df = pd.merge(health_df, timeline_df, left_index=True, right_index=True, how='outer')
        joseph_df['person'] = joseph_df['person'].fillna('Joseph')

        print("\nCombined Joseph DataFrame Preview:")
        print(joseph_df.head())
        print(joseph_df.info())
        return joseph_df

    except FileNotFoundError:
        print(f"Error: Joseph timeline file not found at {JOSEPH_TIMELINE_PATH}")
        return health_df if 'health_df' in locals() else pd.DataFrame()
    except Exception as e:
        print(f"An error occurred processing Joseph's timeline data: {e}")
        return health_df if 'health_df' in locals() else pd.DataFrame()


def process_jackson_data():
    print("\n--- Processing Jackson's Data ---")
    try:
        df = pd.read_csv(JACKSON_DATA_PATH)
        print(f"Initial Jackson data shape: {df.shape}")
        df['calendarDate'] = pd.to_datetime(df['calendarDate'])
        numeric_cols_to_fill = ['moderateIntensityMinutes_x', 'vigorousIntensityMinutes_x',
                                'moderateIntensityMinutes_y', 'vigorousIntensityMinutes_y',
                                'totalSteps', 'deepSleepSeconds', 'lightSleepSeconds']
        for col in numeric_cols_to_fill:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        agg_funcs = {
            'person': 'first',
            'restingHeartRate': 'mean',
            'averageStressLevel': 'mean',
            # No HRV, skipping
            'events.academic': lambda x: list(x.dropna()),
            'events.work': lambda x: list(x.dropna()),
            'events.activities': lambda x: list(x.dropna()),
            'moderateIntensityMinutes_x': 'sum',
            'vigorousIntensityMinutes_x': 'sum',
            'duration_x': 'sum',
            'totalSteps': 'first',
            'deepSleepSeconds': 'first',
            'lightSleepSeconds': 'first',
        }

        agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}

        daily_df = df.groupby('calendarDate').agg(agg_funcs).reset_index()
        print(f"Jackson data shape after grouping by date: {daily_df.shape}")

        jackson_timeline_features = {}
        for _, row in daily_df.iterrows():
            date_obj = row['calendarDate']

            event_count = 0
            total_duration = 0
            academic_count = 0
            assignment_count = 0
            assignment_difficulty_sum = 0
            exercise_duration_weighted = 0
            early_late_duration = 0

            all_academic_events_str = row.get('events.academic', [])
            all_work_events_str = row.get('events.work', [])
            all_activity_events_str = row.get('events.activities', [])

            academic_list = []
            for item in all_academic_events_str:
                parsed = safe_literal_eval(item)
                if isinstance(parsed, list):
                    academic_list.extend(parsed)
                elif isinstance(parsed, dict):
                    academic_list.append(parsed)

            for event in academic_list:
                 if not isinstance(event, dict): continue
                 event_count += 1
                 academic_count += 1
                 duration = 0
                 start_dt, end_dt = pd.NaT, pd.NaT
                 time_str = event.get('time')
                 if time_str:
                      start_dt, end_dt = parse_time_range(time_str, date_obj)
                      duration = calculate_duration_from_dt(start_dt, end_dt)
                 elif 'duration_minutes' in event:
                      duration = event.get('duration_minutes', 0)

                 total_duration += duration

                 if not pd.isna(start_dt) and (start_dt.time() < time(9, 0) or (not pd.isna(end_dt) and end_dt.time() > time(21, 0) and end_dt.date() == start_dt.date()) or (not pd.isna(end_dt) and end_dt.time() <= time(9,0) and end_dt.date() > start_dt.date())):
                     early_late_duration += duration

                 task = event.get('task', '').lower()
                 if 'assignment' in task or 'quiz' in task or 'present' in task or 'project' in task or 'reading' in task:
                     assignment_count += 1
                     assignment_difficulty_sum += event.get('difficulty', 0)

            work_list = []
            for item in all_work_events_str:
                parsed = safe_literal_eval(item)
                if isinstance(parsed, list):
                    work_list.extend(parsed)
                elif isinstance(parsed, dict):
                    work_list.append(parsed)

            for event in work_list:
                if not isinstance(event, dict): continue
                event_count += 1
                duration = event.get('hours', 0) * 60
                exertion = event.get('exertion', 1)
                total_duration += duration

            activity_list = []
            for item in all_activity_events_str:
                parsed = safe_literal_eval(item)
                if isinstance(parsed, list):
                    activity_list.extend(parsed)
                elif isinstance(parsed, dict):
                    activity_list.append(parsed)

            for event in activity_list:
                if not isinstance(event, dict): continue
                event_count += 1
                duration = event.get('duration_minutes', 0)
                total_duration += duration

            garmin_activity_duration_min = 0
            if 'duration_x' in row and pd.notna(row['duration_x']):
                 garmin_activity_duration_min = row['duration_x'] / 1000 / 60 # ms to min

            mod_mins = row.get('moderateIntensityMinutes_x', 0)
            vig_mins = row.get('vigorousIntensityMinutes_x', 0)
            if garmin_activity_duration_min > 0:
                 avg_intensity_weight = ((mod_mins * 5) + (vig_mins * 10)) / (mod_mins + vig_mins) if (mod_mins + vig_mins) > 0 else 1
                 exercise_duration_weighted += garmin_activity_duration_min * avg_intensity_weight

            jackson_timeline_features[date_obj] = {
                'event_count': event_count,
                'total_duration': total_duration,
                'academic_count': academic_count,
                'assignment_count': assignment_count,
                'assignment_difficulty_sum': assignment_difficulty_sum,
                'exercise_duration_weighted': exercise_duration_weighted,
                'early_late_duration': early_late_duration
            }

        timeline_df = pd.DataFrame.from_dict(jackson_timeline_features, orient='index')
        timeline_df.index.name = 'calendarDate'

        jackson_df = daily_df[['calendarDate', 'person', 'restingHeartRate', 'averageStressLevel']].copy()
        jackson_df.rename(columns={'averageStressLevel': 'stress'}, inplace=True)
        jackson_df['stress'] = jackson_df['stress'].apply(lambda x: x if pd.isna(x) or x >= 0 else np.nan)

        jackson_df = pd.merge(jackson_df, timeline_df, on='calendarDate', how='outer')
        jackson_df.set_index('calendarDate', inplace=True)
        jackson_df = jackson_df.sort_index()

        print("\nCombined Jackson DataFrame Preview:")
        print(jackson_df.head())
        print(jackson_df.info())
        return jackson_df

    except FileNotFoundError:
        print(f"Error: Jackson data file not found at {JACKSON_DATA_PATH}")
        return pd.DataFrame()
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred processing Jackson's data: {e}")
        print(traceback.format_exc()) # Print full traceback for debugging
        return pd.DataFrame()


if __name__ == "__main__":
    spencer_data = process_spencer_data()
    joseph_data = process_joseph_data()
    jackson_data = process_jackson_data()

    spencer_data = add_google_history_features(spencer_data, 'Spencer')
    joseph_data = add_google_history_features(joseph_data, 'Joseph') 
    jackson_data = add_google_history_features(jackson_data, 'Jackson')

    start_date = '2025-01-20'
    end_date = '2025-04-20'
    spencer_data = spencer_data[(spencer_data.index >= start_date) & (spencer_data.index <= end_date)]
    joseph_data = joseph_data[(joseph_data.index >= start_date) & (joseph_data.index <= end_date)]
    jackson_data = jackson_data[(jackson_data.index >= start_date) & (jackson_data.index <= end_date)]
    
    print("\n--- Data Loading Complete ---")
    print(f"Spencer data shape: {spencer_data.shape}")
    print(f"Joseph data shape: {joseph_data.shape}")
    print(f"Jackson data shape: {jackson_data.shape}")


    print("\n--- Saving Processed Data to CSV Files ---")
    try:
        if not spencer_data.empty:
            spencer_csv_path = os.path.join(BASE_DATA_PATH, 'spencer_processed.csv')
            spencer_data.to_csv(spencer_csv_path, index=True)
            print(f"Spencer's data saved to: {spencer_csv_path}")
        else:
            print("Spencer DataFrame is empty, not saving CSV.")

        if not joseph_data.empty:
            joseph_csv_path = os.path.join(BASE_DATA_PATH, 'joseph_processed.csv')
            joseph_data.to_csv(joseph_csv_path, index=True)
            print(f"Joseph's data saved to: {joseph_csv_path}")
        else:
            print("Joseph DataFrame is empty, not saving CSV.")

        if not jackson_data.empty:
            jackson_csv_path = os.path.join(BASE_DATA_PATH, 'jackson_processed.csv')
            jackson_data.to_csv(jackson_csv_path, index=True)
            print(f"Jackson's data saved to: {jackson_csv_path}")
        else:
             print("Jackson DataFrame is empty, not saving CSV.")
    except Exception as e:
        print(f"Error saving CSV files: {e}")










    # Example: Combine all data into one DataFrame (optional)
    # combined_df = pd.concat([spencer_data.reset_index(),
    #                          joseph_data.reset_index(),
    #                          jackson_data.reset_index()], ignore_index=True)
    # print("\n--- Combined DataFrame ---")
    # print(combined_df.head())
    # print(combined_df.info())
    # print(combined_df.groupby('person').describe())

    # --- Next Steps (Placeholder) ---
    # Here you would proceed with:
    # 1. Handling Missing Data (NaNs) - imputation or removal
    # 2. Feature Engineering (e.g., creating interaction terms, lag features)
    # 3. Statistical Correlation Analysis
    # 4. Machine Learning Model Training & Evaluation
    # print("\nPlaceholder for further analysis...")