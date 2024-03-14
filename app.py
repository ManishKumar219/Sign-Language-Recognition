from flask import Flask, render_template, Response
import cv2

import cv2, pickle
import tensorflow as tf
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread
import numpy as np
import mediapipe as mp

# Initialize Flask app
app = Flask(__name__)

# Initialize the Audio Module
engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize the Model
model = load_model(r"artifacts/training/model.h5")

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils
mpDrawingStyle = mp.solutions.drawing_styles

def model_prediction(model, landmarks):
    pred_prob = model.predict(landmarks)
    pred_class = np.argmax(pred_prob)
    return pred_class, max(max(pred_prob))


def process_image(frame):
    # frame = frame.to_ndarray(format="bgr24")
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frameRGB

def get_landmarks(result):
    lmsList = []
    # result = hands.process(frameRGB)
    if result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]
        for lm in handLms.landmark:
            # h, w, c = frameRGB.shape
            lmsList.append(lm.x)
            lmsList.append(lm.y)
            lmsList.append(lm.z)
        lmsList = [lmsList]
        lmsList = np.array(lmsList)
    return lmsList


def say_text(text):
	if not is_voice_on:
		return
	while engine._inLoop:
		pass
	engine.say(text)
	engine.runAndWait()

def get_text_from_database(pred_class):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    command = "SELECT label FROM gesture WHERE label_index=" + str(pred_class)
    cursor.execute(command)
    for row in cursor:
        return row[0]

def get_pred_from_landmarks(lms):
    text = ""
    pred_class, pred_prob = model_prediction(model, lms)
    # print(pred_class, pred_prob)
    if (pred_prob * 100 > 60):
        text = get_text_from_database(pred_class)
    return text

x, y, w, h = 300, 100, 300, 300
is_voice_on = True




# Function to capture video frames from the camera
def generate_frames():
    cap = cv2.VideoCapture(0)
    text = ""
    word = ""
    count_same_frame = 0

    while True:
        # Capture frame-by-frame
        success, frame = cap.read()

        if not success:
            break
        else:
            frameRGB = process_image(frame)
            result = hands.process(frameRGB)
            old_text = text
            if result.multi_hand_landmarks:
                # handLms = result.multi_hand_landmarks[0]
                # mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, mpDrawingStyle.get_default_hand_landmarks_style(),
                #                             mpDrawingStyle.get_default_hand_connections_style())
                lms = get_landmarks(result)
                text = get_pred_from_landmarks(lms)
                if(old_text == text):
                    count_same_frame += 1
                else:
                    count_same_frame = 0
                    
                if count_same_frame > 10:
                    if len(text) == 1:
                        Thread(target=say_text, args=(text, )).start()
                    word = word + text
                    if word.startswith('I/Me '):
                        word = word.replace('I/Me ', 'I ')
                    elif word.endswith('I/Me '):
                        word = word.replace('I/Me ', 'me ')
                    count_same_frame = 0

            else:
                if(word!=''):
                    Thread(target=say_text, args=(word, )).start()
                text = ""
                word = ""

            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
            cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
            cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
            if is_voice_on:
                cv2.putText(blackboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
            else:
                cv2.putText(blackboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
            
            res = np.hstack((frame, blackboard))
            # cv2.imshow("Recognizing gesture", res)

            # Encode the frame into JPEG format
            ret, buffer = cv2.imencode('.jpg', res)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign_to_text')
def sign_to_text():
    return render_template('sign_to_text.html')

@app.route('/text_to_sign')
def text_to_sign():
    return render_template('text_to_sign.html')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)