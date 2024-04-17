import cv2
import numpy as np
import mediapipe as mp
import joblib
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
socketio = SocketIO(app)

model = joblib.load('rfctrained_model4.joblib')

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
predicted_class = ""
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def preprocess(l):
    newdffirstl = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 10, 11, 18, 19, 20, 21, 22, 23, 18, 19, 26, 27, 28, 29, 30, 31, 26, 27, 34, 35, 36, 37, 38, 39, 0, 1]
    newdfsecondl = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 34, 35]
    dfl = []
    for i, j in zip(newdffirstl, newdfsecondl):
        dfl.append(l[i]-l[j])
    return dfl

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y])
                preprocessed = preprocess(landmarks)
                predicted_class = model.predict([preprocessed])[0]
                socketio.emit('predicted_class', {'class_label': predicted_class})
                # print(predicted_class)
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
