import pandas as pd
from glob import glob
import re

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_acc = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)

single_file_gyr = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)


# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")
print(files)
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
# data_path = "../../data/raw/MetaMotion/"

file = files[0]
filename = file.replace(
    "\\", "/"
)  # Replace double backslashes with single forward slashes

participant = filename.split("/")[-1].split("-")[
    0
]  # Extract participant from filename (A,B,.. etc)
exercise = filename.split("/")[-1].split("-")[1]  # Extract exercise
category = filename.split("/")[-1].split("-")[2]  # Extract heavy/light category
set_number = re.sub(r"\D", "", category)  # Remove all non-digit characters
category = re.sub(
    r"\d+", "", category
)  # Remove numerical values using regular expressions

df = pd.read_csv(file)
df["participant"] = participant
df["exercise"] = exercise
df["category"] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()  # creation of empty df
gyr_df = pd.DataFrame()

acc_set = 1  # used to create unique identifier
gyr_set = 1

for file in files:
    filename = file.replace(
        "\\", "/"
    )  # Replace double backslashes with single forward slashes
    participant = filename.split("/")[-1].split("-")[
        0
    ]  # Extract participant from filename (A,B,.. etc)
    exercise = filename.split("/")[-1].split("-")[1]  # Extract exercise
    category = filename.split("/")[-1].split("-")[2]  # Extract heavy/light category
    # set_number = re.sub(r"\D", "", category)  # Remove all non-digit characters
    category = re.sub(r"\d+|_MetaWear_", "", category)
    # Remove numerical values and "_MetaWear_" using regular expressions

    df = pd.read_csv(file)
    df["participant"] = participant
    df["exercise"] = exercise
    df["category"] = category

    if "Accelerometer" in file:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat(
            [acc_df, df]
        )  # builds the acc_df if file name contains "Accelerometer"

    if "Gyroscope" in file:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])  # builds gyr_df

    """gyr_df contains more rows as the gyroscope was recording data ata higher frequency (more measurements per)"""


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.info()
"""
    0 epoch (ms)    23578 non-null  int64  
    1 time (01:00)  23578 non-null  object 
"""
pd.to_datetime(df["epoch (ms)"], unit="ms")

# create index datetime

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")


def read_data_from_files(files):

    acc_df = pd.DataFrame()  # creation of empty df
    gyr_df = pd.DataFrame()

    acc_set = 1  # used to create unique identifier
    gyr_set = 1

    for file in files:
        filename = file.replace(
            "\\", "/"
        )  # Replace double backslashes with single forward slashes
        participant = filename.split("/")[-1].split("-")[
            0
        ]  # Extract participant from filename (A,B,.. etc)
        exercise = filename.split("/")[-1].split("-")[1]  # Extract exercise
        category = filename.split("/")[-1].split("-")[2]  # Extract heavy/light category
        category = re.sub(r"\d+|_MetaWear_", "", category)
        # Remove numerical values and "_MetaWear_" using regular expressions

        df = pd.read_csv(file)
        df["participant"] = participant
        df["exercise"] = exercise
        df["category"] = category

        if "Accelerometer" in file:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat(
                [acc_df, df]
            )  # builds the acc_df if file name contains "Accelerometer"

        if "Gyroscope" in file:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])  # builds gyr_df

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

merged_df = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

merged_df.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

# equation: time = 1/frequency(Hz)

aggregation_method = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}


# merged_df[:100].resample(rule="200ms").apply(aggregation_method)
"""the above line is fine for small amounts of rows but if applied to the whole merged_df would create an unreasonable amount of null rows as we are resampling between time series data spanning a week"""

days = [g for n, g in merged_df.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(aggregation_method).dropna() for df in days]
)

data_resampled.info()
data_resampled["set"] = data_resampled["set"].astype("int")

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")