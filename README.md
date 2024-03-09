# Utilising Machine Learning: Multiclassification of sensor data recorded during various exercises

## Overview
This project aims to develop a Python-based excercise tracker/classfier using accelerometer and gyroscope data from a Metamotion sensor. The goal is to create a machine learning model capable of classifying various barbell exercises and counting repetitions using data obtained from the sensor.

The provided dataset, available in the `metamotion.zip` file ('data/raw'), contains gyroscope and accelerometer data for heavy and light barbell movements. Data is collected for all three axes for both gyroscopes and accelerometers, including rest data. It contains a weeks worth of data from various people all doing the same exercises.

Once fully functional, this project could serve as the foundation for a versatile fitness app. Personal trainers could use it to monitor clients remotely, while individuals with compatible devices (e.g., Apple Watch) could track workouts effortlessly.

The Metamotion sensor provides comprehensive data, including gyroscope, accelerometer, magnetometer, sensor fusion, barometric pressure, and ambient light.

![Metamotion Sensor](https://mbientlab.com/wp-content/uploads/2021/02/Board4-updated.png)

## Contents
1. [Overview](#overview)
2. [Data Collection](#data-collection)
3. [Python Scripts](#python-scripts)
4. [Data Visualization](#data-visualization)
5. [Outlier Detection & Management](#outlier-detection--management)
6. [Feature Engineering](#feature-engineering)



## Data Collection
Sensor data is collected during workouts to capture barbell movement and orientation, including rest periods. Data was collected across a period of one week by a number of participants.

its important to note that the gyroscope recorded data at a greater frequency than the accelerometer.

Accelerometer:    12.500HZ

Gyroscope:        25.000Hz

### Barbell Exercises
- Barbell Bench Press
- Barbell Back Squat
- Barbell Overhead Press
- Deadlift
- Barbell Row

## Python Scripts
- `make_dataset.py` : (`src/data`) - Contains code used to extract and transform the original raw data.
- `visualize.py` : (`src/visualization`) - Includes code for generating various visualizations from the preprocessed sensor data.
- `remove_outliers.py` : (`src/features`) - Implements outlier detection methods and removes outliers from the dataset.
- `build_features.py` : (`src/features`) - Implements feature engineering steps such as dealing with missing values, calculating set duration, applying a Butterworth lowpass filter, conducting principal component analysis (PCA), computing sum of squares attributes, and potentially other feature engineering tasks such as temporal abstraction, frequency features, dealing with overlapping windows, clustering, and exporting the dataset.
- `DataTransformation.py` : (`src/features`) - Provides functions and classes for data transformation tasks such as low-pass filtering and principal component analysis.


## Machine Learning Model
We will be using supervised learning techniques, as we have both structured and unstructured data. The goal is to create a multiclass classification model to predict which exercise is being done or if the participant is resting. 

## Data Extraction & Transformation:  `make_dataset.py` 
<i>filepath : `src/data/make_dataset.py`</i>

### Extracting information from filenames:

The gyroscope and accelerometer raw data files within the MetaMotion folder are named as follows:

```
# Accelerometer:
"../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"

#Gyroscope:
../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv
```

The file naming convention prominently includes several key components: a participant identifier (A), an exercise label (bench), a category descriptor (heavy), and supplementary details like RPE (rate of perceived exertion).

To initiate the extraction of information from the file path and name, refer to the code snippet below:

```
files = glob("../../data/raw/MetaMotion/*.csv")
print(files)
```
This returns:
```
['../../data/raw/MetaMotion\\A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv',...]
```
Here we have already encountered my first issue, the filepath returned now contains `\\` instead of `/`
which will cause an issue when trying to .split . 

```
for file in files:
    
    filename = file.replace("\\", "/")  
```
The simple replacement above will now allow us to split the filename and extract participant, exercise and category.
The *re* regular expressions module was used to remove the `_MetaWear_` and any numerical values:

```
    participant = filename.split("/")[-1].split("-")[0]    
    exercise = filename.split("/")[-1].split("-")[1]  
    category = filename.split("/")[-1].split("-")[2]  
    category = re.sub(r"\d+|_MetaWear_", "", category) 
```
`participant`, `exercise`, `category` have been extracted.

### Dataframe creation:

By reading the file into a dataframe we can then create appropriately named columns to store the extracted variables:


```
    df = pd.read_csv(file)
        df["participant"] = participant
        df["exercise"] = exercise
        df["category"] = category
```

The initialization of two empty dataframes and two variables depicted below serves the purpose of concatenating all dataframes into the corresponding ones and generating a unique identifier, "set," for each data file. These elements will be pre-set before the aforementioned for loop

```
acc_df = pd.DataFrame() 
gyr_df = pd.DataFrame()

acc_set = 1  
gyr_set = 1
```

Based on the filename, we generate the set column in the dataframe and populate it using the corresponding set variable (`acc_set` or `gyr_set`). Subsequently, we concatenate this updated dataframe with the existing Accelerometer (`acc_df`) or Gyroscope (`gyr_df`) dataframes.

Throughout the loop iterating over all files, these two dataframes will continuously update and expand.

```
    if "Accelerometer" in file:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df]) 

    if "Gyroscope" in file:
        df["set"] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df])
```

### Formatting, pruning and merging dataframes:

Creating a new `epoch (ms)` column, assigning as the index and converting from Unix to datetime in the units 'ms'.

```
acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")
```

Pruned unnecessary columns:
```
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]
```

Merging `acc_df` and `gyr_df` into one dataframe `merged_df`:

```
merged_df = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
```
Renaming columns so the dataframe is easier to understand at a glance:

```
merged_df.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "label",
    "exercise",
    "category",
    "set",
]
```

### Resampling Time-Series data:
After merging our dataframes, we observe that numerous rows exclusively contain gyroscope data, others exclusively accelerometer data, while only a few include both.

This disparity arises from the gyroscope sensor's higher recording frequency compared to the accelerometer sensor. Consequently, instances of both sensors capturing data simultaneously are rare.

To enhance the completeness of our dataset, we propose resampling the data using aggregation methods such as mean for numerical values and last for non-numeric ones. This approach leverages the time-series nature of our data, enabling us to consolidate and fill in missing values effectively.

```
aggregation_method = {
    "acc_y": "mean",
    "acc_x": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "label": "last",
    "exercise": "last",
    "category": "last",
    "set": "last",
}
```

Before applying the resampling methods described above, it's crucial to acknowledge that we're working with time-series data. Resampling the entire `merged_df` without consideration would lead to an excessive number of null rows. Given that the data spans a week in terms of time series, and considering it was collected only for short durations each day, a direct resampling approach would generate impractical amounts of missing values.

Instead of resampling the entire dataset at once, the dataset is split into smaller groups based on daily intervals. This reduces the computational burden of the resampling operation:

```
days = [g for n, g in merged_df.groupby(pd.Grouper(freq="D"))]
```
Resampling at a frequency of 200 milliseconds appeared to retain a significant number of rows and produce a more comprehensive dataframe named `data_resampled`.

```
data_resampled = pd.concat([df.resample(rule="200ms").apply(aggregation_method).dropna() for df in days])
```

#### Formatting the 'set' column:

```
data_resampled["set"] = data_resampled["set"].astype("int")
```

### Exporting `data_resampled` dataframe to .pkl:

Exporting data to a pickle file is a versatile and efficient solution for storing and exchanging serialized data in Python, offering advantages in terms of serialization efficiency, data preservation, compatibility, ease of use, and support for compression.

```
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
```
## Data Visualization
filepath : `src\visualization\visualize.py`

In this section, we explore sensor data visualizations to gain crucial insights into exercise patterns and participant behaviour, enhancing our machine learning model. These visuals offer a comprehensive understanding of the data dynamics,

### Comparing Medium vs. Heavy Sets
We compare medium vs. heavy sets for a specific exercise, such as squats:

```category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()
fig, ax = plt.subplots()
category_df.groupby(['category'])['acc_y'].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel ("samples")
plt.legend()
plt.savefig("../../reports/figures/Squat_A_Heavy_Medium.png")
```
Resulting plot:
![Squat A Heavy Medium](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/Squat_A_Heavy_Medium.png)
<small><i>Figure 1: Participant A Squats - Heavy/Medium - No of Samples vs Accelerometer (y-axis)</i></small>

Figure 1 indicates that Participant A exhibited reduced acceleration along the y-axis when training with a heavy weight compared to a medium weight. While this outcome was anticipated, it's reassuring to see that our data aligns with this expectation.

### Comparison of Accelerometer and Gyroscope measurements per participant per exercise:

Here's where things get interesting:


```
for label in labels:
    for participant in participants:
        all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
            all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
            all_axis_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])

            ax[0].legend(loc='upper center', bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,shadow=True)
            ax[1].legend(loc='upper center', bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,shadow=True)
            ax[1].set_xlabel('samples')

            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()
```

The for loop generates a series of figures which we can help with:

Comparative Analysis: By comparing accelerometer and gyroscope data between different exercises and participants, we can identify patterns and variations in movement. This aids in selecting relevant features for the model and understanding how different factors influence the sensor data.

Insights into Movement: Understanding movement patterns and characteristics helps in feature engineering. By extracting relevant features from the sensor data, we can provide the model with meaningful input that captures important aspects of exercise performance.

Anomaly Detection: Detecting anomalies in the sensor data helps in data preprocessing and quality control. By identifying and addressing irregularities, we ensure that the input data for the machine learning model is clean, accurate, and representative of typical exercise performance. This enhances the model's ability to generalize and make accurate predictions.

![Participant B Bench](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/Bench%20(B).png)
<small><i>Figure 2: Participant B Bench Press - plot displaying accelerometer and gyroscope data in all three axis (x,y,z).</i></small>

![Participant D Bench](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/Bench%20(D).png)
<small><i>Figure 3: Participant D Bench Press - plot displaying accelerometer and gyroscope data in all three axis/planes (x,y,z).</i></small>


In both <i>Figure 2</i> and <i>Figure 3</i>, it's noticeable that despite both participants performing the same exercise, there are considerable differences in the gyroscope data and acc_y plots. These disparities may stem from variations in bench press execution, such as differences in form, rep control, and speed. However, across all Bench Plots from all participants, there's a consistent observation of minimal acceleration in the x and z plane and maximal acceleration in the y plane.

## Outlier Detection & Management

filepath : `src\features\remove_outliers.py`

We visualise the outliers from our `data/interim/01_data_processed.pkl` file, using box plots and histograms to understand their distribution across different exercises and participants.

To detect outliers, we implement three different methods: interquartile range (IQR), Chauvenet's criterion, and Local Outlier Factor (LOF). After evaluating these methods.

![IQR](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/IQR.png)
![Chauvenet](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/Chauvenet.png)
![LOF](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/LOF.png)
<small><i>Figure 4: Top (IQR Outlier Detection), Middle (Chauvenet Outlier Detection), Bottom (LOF Outlier Detection)</i></small>

Opting for Chauvenet's criterion for outlier detection, we leverage its assumption of a normal distribution, which aligns well with our data characteristics. Additionally, it tends to flag fewer outliers, but those identified are more pertinent to our analysis.

Chauvenet's criterion identifies outliers based on the assumption of a normal distribution, making it suitable for our sensor data analysis. We apply this method to each sensor's data columns, marking outliers and subsequently replacing them with NaN values. This approach ensures that the machine learning model isn't skewed by anomalous data points, leading to more robust and accurate predictions.

Finally, we export the cleaned dataset with outliers removed, ready for further preprocessing and model development.

see file: `data\interim\02_outliers_removed_chauvenets.pkl`

## Feature engineering:
This section outlines the feature engineering pipeline used to preprocess the sensor data for exercise classification. The analysis involves various stages, including handling missing values, filtering, dimensionality reduction, and extracting relevant features.

### Dealing with missing values:
Missing values in the dataset are handled through linear interpolation to ensure continuity in the time series data using the for loop below:

```
for col in predictor_columns:
    df[col] = df[col].interpolate()
```

Where:
```
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])
```
Interpolation/Imputation Visualized:

![Set 35 gyr_y plot (before imputation)](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/Set%2035%20gyr_y%20plot%20(before%20imputation).png)
<i>Figure 5: gyr_y data plot for set 35 Overhead Press, pre interpolation.</i>

![Set 35 gyr_y plot (after imputation)](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/Set%2035%20gyr_y%20plot%20(after%20imputation).png)
<i>Figure 6: gyr_y data plot for set 35 Overhead Press, post interpolation.</i>

### Calculating Set duration: 
The duration of each exercise set is calculated to provide insights into the length of time spent on each exercise.

```
for s in df["set"].unique():
    start = df[df["set"]== s].index[0]
    stop = df[df["set"]== s].index[-1]
    
    duration = stop - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds
    
duration_df = df.groupby(["category"])["duration"].mean()
duration_df.iloc[0] / 5 #5 heavy reps
duration_df.iloc[1] / 10 #10 medium reps
```

The code above computes the duration of each exercise set by subtracting the start time from the stop time. It then calculates the average duration for each exercise category. This helps in understanding the typical time taken for sets and repetitions, facilitating performance assessment and training optimization.

Returning: 

```
category
heavy       14.743501
medium      24.942529
sitting     33.000000
standing    39.000000
Name: duration, dtype: float64
```

### Application of Butterworth Low Pass filter:

```
df_lowpass = df.copy()
LowPass = LowPassFilter()

s_freq = 1000 / 200 #200 ms : 5 instances per second
cutoff_freq = 1.3

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col , s_freq,cutoff_freq, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
```

The code snippet provided is utilizing the LowPassFilter class from the DataTransformation module (`DataTransformation.py`) to perform the lowpass filtering operation on the accelerometer and gyroscope data.

![Low Pass Filter](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/Low_Pass_Filter.png)
<i>Figure 7: Application of Butterworth Low Pass Filter.</i>

The code applies a Butterworth lowpass filter to the accelerometer and gyroscope data in order to remove high-frequency noise while preserving the underlying signal trends. 

The cutoff frequency of the filter is chosen to attenuate frequencies above a certain threshold, effectively smoothing out rapid fluctuations in the data. This is particularly useful for motion sensor data, where high-frequency noise can obscure meaningful patterns and introduce inaccuracies in analysis. 

The filtered data (`df_lowpass`) is then used for further analysis, ensuring that subsequent calculations and visualizations are based on a cleaner representation of the original sensor measurements.

### Application of Principal Component Analysis (PCA):

The application of PCA on `df_lowpass` begins with determining the explained variance for each principal component, providing insight into how much of the original data's variability is captured by each component.

The code snippet below is utilizing the PrincipalComponentAnalysis class from the DataTransformation module (`DataTransformation.py`)

```
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pca_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
```

Using the elbow method, the optimal number of principal components to retain is identified <i>see figure 8</i>, balancing dimensionality reduction with information retention.

```
plt.figure(figsize = (10,10))
plt.plot(range(1, len(predictor_columns) + 1), pca_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()
```

![elbow method](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/elbow_method.png)
<i>Figure 8: Plot identifies the optimal number of components as 3.</i>

```
df_pca = PCA.apply_pca(df_pca,predictor_columns,3)
```

Subsequently, PCA is applied to transform the dataset into a lower-dimensional space while preserving most of its variability.

```
subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()
```

This transformation allows for the representation of the data along orthogonal directions, facilitating efficient computation and visualization. Finally, the transformed data is visualized in <i>Figure 9 and 10</i> to understand the patterns and structure captured by the principal components, aiding in further analysis and interpretation of the dataset.

![PCA](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/PCA.png)
<i>Figure 9: Visualization of Principal Components for Set 35, Medium Overhead Press.</i>

![PCA Heavy Bench](https://raw.githubusercontent.com/NadeemDin/ML-Sensor-Exercise-Multiclassification/main/reports/figures/PCA_Heavy_Bench.png)
<i>Figure 10: Visualization of Principal Components for Set 40, Heavy Bench.</i>





