import dev
import os 
import pandas as pd
import json
from datetime import datetime, timedelta

def convert_garmin_time(seconds):
    return garmin_epoch + timedelta(seconds=seconds)

if __name__ == "__main__":
    directory = os.getcwd() + "/jackson-data"
    garmin_epoch = datetime(1989, 12, 31)
    # jd = dev.GetJacksonData(directory=directory)
    # jd.get_garmin_data()
    file_path = directory + "/" + "garmin_data/" + "summarizedActivities.json"
    with open(file_path) as f:
        raw_data = json.load(f)[0]['summarizedActivitiesExport']
    # df.to_csv(directory + "/garmin_data/" + "temp.csv")
    df = pd.DataFrame(raw_data)
    df['datetime'] = df['beginTimestamp'].apply(lambda x: datetime.utcfromtimestamp(x / 1000))
    print(df.columns)
    # datetime, activityType, name, sportType, avgHr, maxHr, calories, bmrCalories, duration, moderateIntensityMinutes, vigorousIntensityMinutes


