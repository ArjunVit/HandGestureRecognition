# Hand Gesture Recognition

This repository contains code for training a machine learning model to recognize hand gestures from videos and a Python script to predict hand gestures in real-time using a webcam.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries: mediapipe, joblib, Flask, scikit-learn

### Installation

1. Clone this repository to your local machine:
   ```
   git clone https://github.com/ArjunVit/hand-gesture-recognition.git
   ```
2. Install the required Python libraries:
   ```
   pip install mediapipe joblib Flask scikit-learn
   ```

## Usage

### Training the Model

1. Place your training videos in the `videos` folder. Each class should have its own folder containing video clips of that class.
2. Run the `main.ipynb` notebook to train the machine learning model. This notebook will preprocess the video data and train the model using scikit-learn.

### Real-time Prediction

1. Run the `predict.py` script to start the Flask server:
   ```
   python predict.py
   ```
2. Open a web browser and navigate to `http://localhost:5000` to view the real-time prediction dashboard.
3. The webcam feed will be displayed along with the detected hand landmarks. The corresponding buttons will be highlighted based on the predicted hand gesture.
