# IrisPupilRatioDetect
"Real-time Iris to Pupil Ratio Detection: Implementation for Live Monitoring and Analysis."

This script utilizes OpenCV and Mediapipe to detect facial landmarks, extract iris and pupil regions, and compute the iris-to-pupil ratio in real-time using a webcam. It processes captured frames, applies contour detection for pupil estimation, and saves annotated images for further analysis.  

Usage  
Once executed, the script opens a webcam feed where users can capture a frame for analysis. It detects eye landmarks, calculates the iris and pupil radius, and determines the iris-to-pupil ratio. The results, including computed measurements, are displayed in the terminal.  

Output 
Annotated images with detected iris and pupil are saved as `left_eye_annotated_iris.png` and `right_eye_annotated_iris.png`.  
The iris-to-pupil ratio is printed in the terminal.  
The extracted data can be utilized for biometric analysis, vision research, or medical diagnostics.
