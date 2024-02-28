# Machine Learning Fitness Tracker Project

## Overview
This project aims to develop a Python-based fitness tracker using accelerometer and gyroscope data from a Metamotion sensor. The goal is to create a machine learning model capable of classifying various barbell exercises and counting repetitions.

The provided dataset, available in the `metamotion.zip` file, contains gyroscope and accelerometer data for heavy and light barbell movements. Data is collected for all three axes for both gyroscopes and accelerometers, including rest data.

Once fully functional, this project could serve as the foundation for a versatile fitness app. Personal trainers could use it to monitor clients remotely, while individuals with compatible devices (e.g., Apple Watch) could track workouts effortlessly.

The Metamotion sensor provides comprehensive data, including gyroscope, accelerometer, magnetometer, sensor fusion, barometric pressure, and ambient light.

![Metamotion Sensor](https://mbientlab.com/wp-content/uploads/2021/02/Board4-updated.png)

## Data Collection
Sensor data is collected during workouts to capture barbell movement and orientation, including rest periods.

## Python Scripts


## Machine Learning Model
We will be using supervised learning techniques, as we have both structured and unstructured data. The goal is to create a multiclass classification model to predict which exercise is being done or if the participant is resting. 

### Barbell Exercises
- Barbell Bench Press
- Barbell Back Squat
- Barbell Overhead Press
- Deadlift
- Barbell Row


