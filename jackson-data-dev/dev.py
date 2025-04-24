import pandas as pd
import numpy as np
import os
from pathlib import Path
from dateutil import parser

"""
    Read in, clean and get the data I need from the directories (not pushing)
"""

class GetJacksonData:
    """
        Give me the directory, then hit run.
    """

    def __init__(self, directory):
        self.directory = directory


    def get_garmin_data(self):
        for name in os.listdir(self.directory + "/garmin_data"):
            if ".DS_Store" not in name:
                tmp_df = self.garmin_helper("garmin_data/" + name)
            

    def get_timeline_data(self):
        return pd.read_json(self.directory + "/timeline/" + "jackson_timeline.json")

    def get_takeout_data(self):
        for root, dirs, files in os.walk(self.directory + "/takeout/Takeout/My Activity"):
            for file in files:
                file_path = os.path.join(root, file)
                print(file_path)

    def put_everything_together(self):
        pass

    def run(self):
        garmin_df = self.get_garmin_data()
        timeline_df = self.get_timeline_data()
        takeout_df = self.get_takeout_data()

