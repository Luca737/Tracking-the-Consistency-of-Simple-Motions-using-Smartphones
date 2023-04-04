# Tracking-the-Consistency-of-Simple-Motions-using-Smartphones
This is most of the source code for my thesis.

## ConstistencyTracker
ConstistencyTracker is the swift app build the in the last section of my Implementation chapter in my thesis. It records the users exercise motion data and analyze it using 3 motion data types, linear acceleration, gravity and rotation rate. For these three types for every axis the repetitions of the user are calculated. From them several statistics are calculated and presented to the user.
This app can be connected to a TrainerHub session running on the same wifi to send the raw motion data and the calculated repetition intervals, as well as additionally starting video capture during the motion recording (eg. for recording a larger data set)

## SensorDataCollect (SensApp)
SensApp is the initial app build for collecting the motion data set for this thesis. It requires a TrainerHub session to run on the same wifi to connect to. When starting a recording SensApp start the motion capture and signals TrainerHub to start the video capture. The resulting data is package into a zip and saved on the host machine running TrainerHub.

## TrainerHub
TrainerHub is written in python and its purpose is to connect to the ConstistencyTracker or SensorDataCollect (SensApp) app for gathering the motion data set acquired in this thesis. It receives the motion data and saves it in a Visual Inspection Tool (VIT, http://doi.org/10.1145/3303772.3303776) compliant way for annotation. When connected to SensApp it allays records video using the first camera of the host device during a motion recording. After the recording is finished it is saved together with the motion data. When connected to ConstistencyTracker, the video capture is optional. In main.py on can change, which app to connect to.
The main data set used in this thesis is included in the testFolder (inside TrainerHub)

## SensorDataAnalysis
This is the collection of tools written for analyzing the motion data and testing out different analysis methods. All analysis methods (in analyze.py) can be tested using rate_analysis.py (in analyze_truth.py) on the the data set provided in the testFolder inside of TrainerHub. The data can also be transformed in various ways using transform.py (or transform_all.py). The transformations are written into a separate JSON file, which is saved inside the zip folder containing the raw motion data and the video capture. This transformed data can than for example be viewed in VIT.

## Requirements.txt
Lists the python packages required for running all python code.
