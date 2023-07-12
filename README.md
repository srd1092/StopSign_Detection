# StopSign_Detection

Run the **cascade_detection.py** file to see the result using the cascade classifier. This should display bounding boxes around the detected stop signs in the given input video file and generates a csv file with the coordinates of the bounding boxes.

Run the **self_model_detection.py** file to see the result of the model that is trained using the GTSRB dataset.
The model design is in the self_training.py file. The model is a classification model with 2 classes (stop sign, other signs). This should display if stop signs are detected in the given input video file.
