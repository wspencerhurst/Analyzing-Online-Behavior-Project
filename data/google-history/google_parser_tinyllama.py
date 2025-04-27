import os
import json
import pandas as pd
from dateutil import parser as date_parser
from dateutil import tz
from datetime import datetime, timedelta
import re
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Local timezone - lmk if you went anywhere crazy
local_tz = tz.gettz("America/New_York")

# Threshold (minutes) for inactivity gap to consider as sleep
SLEEP_INACTIVITY_THRESHOLD = 300  # 5 hours

# Words to ignore in activity content
STOPWORDS = {"watched", "watch", "google", "youtube", "search", "com", "www", "https", "http"}

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load TinyLlama Model
print(f"Loading TinyLlama model onto {device}...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)
print("TinyLlama loaded successfully.")


def read_chrome_history(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for item in data.get("Browser History", []):
        time_usec = item.get("time_usec")
        if time_usec and time_usec > 0:
            dt = datetime.fromtimestamp(int(time_usec) / 1e6, tz=tz.UTC)
            records.append({
                "datetime": dt.astimezone(local_tz),
                "source": "Chrome",
                "title": item.get("title", ""),
                "url": item.get("url", "")
            })
    return pd.DataFrame(records)

def read_youtube_history(watch_path, search_path):
    dfs = []
    for path in [watch_path, search_path]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            times = [{
                "datetime": date_parser.parse(item.get("time")).astimezone(local_tz),
                "source": "YouTube",
                "title": item.get("title", ""),
                "url": item.get("titleUrl", "")
            } for item in data if "time" in item]
            dfs.append(pd.DataFrame(times))
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=["datetime", "source"])
    
def extract_phrases(ngrams, titles, stopwords, top_n=10):
    text = " ".join(titles).lower()
    # Only keep words of at least 4 letters
    words = re.findall(r'\b\w{4,}\b', text)
    words = [w for w in words if w not in stopwords]
    if ngrams == 2:
        grams = zip(words, words[1:])
    if ngrams == 3:
        grams = zip(words, words[1:], words[2:])
    phrases = [" ".join(gram) for gram in grams]
    counter = Counter(phrases)
    return ", ".join([phrase for phrase, count in counter.most_common(top_n)])

def summarize_titles(titles, max_tokens=100):
    try:
        activities = " ".join(titles)
        prompt = (
            "You are categorizing a person's day.\n\n"
            "TASK:\n"
            "- Read the activities.\n"
            "- Group them into broad topics like studying, technology, finance, entertainment.\n"
            "- DO NOT use video titles or video names.\n"
            "- Write 3 to 5 short bullet points.\n"
            "- Each bullet must be a short general category, 2-5 words only.\n"
            "- Example:\n"
            "- Studying for exams\n"
            "- Watching finance news\n"
            "- Gaming tutorials\n\n"
            f"ACTIVITIES:\n{activities}\n\n"
            "SUMMARY (bullet points):"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.4, top_p=0.9)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        summary = generated.replace(prompt, "").strip()
        return summary
    except Exception as e:
        print(f"Warning: Summarization failed. Error: {e}")
        return ""



def rate_stress_level(titles, max_tokens=10):
    try:
        activities = " ".join(titles)
        prompt = (
            "You are estimating how stressful a day was.\n\n"
            "TASK:\n"
            "- Read the activities.\n"
            "- If the activities involve academic studying, work, deadlines, coding, or finance → Rate high stress (7-10).\n"
            "- If the activities are entertainment, gaming, relaxation, casual browsing → Rate low stress (1-3).\n"
            "- If a mix, rate medium (4-6).\n"
            "- Only respond with a single number from 1 to 10. Nothing else.\n"
            "- Example: 7\n\n"
            f"ACTIVITIES:\n{activities}\n\n"
            "STRESS RATING:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.4, top_p=0.9)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        response_text = generated.replace(prompt, "").strip()
        number = re.search(r"\b([1-9]|10)\b", response_text)
        if number:
            return int(number.group(0))
        else:
            return -1
    except Exception as e:
        print(f"Warning: Stress rating failed. Error: {e}")
        return -1





def main():
    chrome_path = os.path.join("Takeout", "chrome", "History.json")
    watch_path = os.path.join("Takeout", "YouTube and YouTube Music", "history", "watch-history.json")
    search_path = os.path.join("Takeout", "YouTube and YouTube Music", "history", "search-history.json")

    chrome_df = read_chrome_history(chrome_path)
    youtube_df = read_youtube_history(watch_path, search_path)

    all_usage = pd.concat([chrome_df, youtube_df], ignore_index=True)
    all_usage = all_usage.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    before_cleaning = len(all_usage)
    all_usage = all_usage[
        (all_usage["datetime"] >= datetime(2000, 1, 1, tzinfo=local_tz)) &
        (all_usage["datetime"] <= datetime(2100, 1, 1, tzinfo=local_tz))
    ]
    after_cleaning = len(all_usage)
    if before_cleaning != after_cleaning:
        print(f"Warning: Dropped {before_cleaning - after_cleaning} invalid timestamps during cleaning.")

    if all_usage.empty:
        print("No activity found.")
        return

    all_usage["time_diff_min"] = all_usage["datetime"].diff().dt.total_seconds() / 60

    sleep_gaps = all_usage[all_usage["time_diff_min"] > SLEEP_INACTIVITY_THRESHOLD].reset_index()

    if sleep_gaps.empty:
        print("No sleep gaps identified.")
        return

    results = {}

    print(all_usage["source"].value_counts())

    for i in tqdm(range(1, len(sleep_gaps)), desc="Processing Days"):
        prev_wake = sleep_gaps.loc[i - 1, "datetime"]
        curr_bed = sleep_gaps.loc[i, "datetime"]

        day = (curr_bed - timedelta(days=1)).date()

        day_activities = all_usage[(all_usage["datetime"] >= prev_wake) & (all_usage["datetime"] <= curr_bed)]

        if not (datetime(2025, 1, 20, tzinfo=local_tz) <= curr_bed <= datetime(2025, 4, 20, 23, 59, 59, tzinfo=local_tz)):
            continue

        if not day_activities.empty:
            earliest = day_activities["datetime"].min().strftime("%H:%M")
            latest = day_activities["datetime"].max().strftime("%H:%M")
            total_count = len(day_activities)
            youtube_count = (day_activities["source"] == "YouTube").sum()
            chrome_count = (day_activities["source"] == "Chrome").sum()

            titles = day_activities["title"].dropna().tolist()
            text = " ".join(titles).lower()
            #words = re.findall(r'\\b\\w{4,}\\b', text)  # only words with 4+ letters
            words = re.findall(r'\b\w{4,}\b', text)
            filtered_words = [word for word in words if word not in STOPWORDS]
            counter = Counter(filtered_words)
            common_words = [word for word, count in counter.most_common(10)]
            summary_keywords = ", ".join(common_words)

            summary_bigrams = extract_phrases(2, titles, STOPWORDS)
            summary_trigrams = extract_phrases(3, titles, STOPWORDS)
            summary = summarize_titles(titles)
            stress_rating = rate_stress_level(titles)



            if day not in results or total_count > results[day]["total_activity_count"]:
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
                    "stress_rating": stress_rating
                }

    df_out = pd.DataFrame([
        {"day": str(day), **vals} for day, vals in sorted(results.items())
    ])
    df_out.to_csv("activity_summary.csv", index=False)
    print("Saved to activity_summary.csv")

if __name__ == "__main__":
    main()
