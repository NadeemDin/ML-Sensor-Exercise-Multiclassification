import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])

#plotter settings:
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
# df.info()
subset = df[df["set"] == 35]["gyr_y"].plot()

for col in predictor_columns:
    df[col] = df[col].interpolate()


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 35]["acc_y"].plot()

duration = df[df["set"]== 1].index[-1] - df[df["set"]== 1].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"]== s].index[0]
    stop = df[df["set"]== s].index[-1]
    
    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds
    
duration_df = df.groupby(["category"])["duration"].mean()
duration_df.iloc[0] / 5 #5 reps
duration_df.iloc[1] / 10 #10 reps

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

s_freq = 1000 / 200 #200 ms : 5 instances per second
cutoff_freq = 1.3

# df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", s_freq,cutoff_freq, order=5)

# subset = df_lowpass[df_lowpass["set"] == 45]
# print(subset["label"][0])

# fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
# ax[0].plot(subset["acc_y"].reset_index(drop=True), label = "raw data")
# ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label = "butterworth filter")
# ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15),fancybox=True,  shadow=True)
# ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15),fancybox=True,  shadow=True)

#overwrites acc and gyr columns with low pass filter applied

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col , s_freq,cutoff_freq, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------




# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------