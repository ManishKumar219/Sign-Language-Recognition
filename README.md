# Chicken-Disease-Classification--Project


<!-- ## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml -->
## Overview

The Sign Language Translator project is a groundbreaking endeavor that seeks to bridge the communication gap between individuals who use sign language and those who do not. Utilizing cutting-edge technology such as computer vision and machine learning, this project aims to revolutionize communication accessibility for the deaf and hard of hearing community.

## Demo
![Example screenshot](./img/demo4.gif)



![Example screenshot](./img/demo2.gif)



![Example screenshot](./img/demo3.gif)


## Technologies and Tools
* Python 
* TensorFlow
* Keras
* OpenCV
* MediaPipe


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/ManishKumar219/Sign-Language-Recognition
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n aslenv python=3.8 -y
```

```bash
conda activate aslenv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```

## Code Examples

````
import cv2, pickle
import tensorflow as tf
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread
import numpy as np
import mediapipe as mp

engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    # img = frame.to_ndarray(format="bgr24")
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return imgRGB

def get_landmarks(result):
    lmsList = []
    # result = hands.process(imgRGB)
    if result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]
        for lm in handLms.landmark:
            # h, w, c = imgRGB.shape
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
MAX_FRAME = 15
SAME_FRAME_CNT = 10

def text_mode(cam):
    global is_voice_on
    text = ""
    word = ""
    count_same_frame = 0
    while True:
        img = cam.read()[1]
        imgRGB = process_image(img)
        result = hands.process(imgRGB)
        old_text = text
        if result.multi_hand_landmarks:
            handLms = result.multi_hand_landmarks[0]
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, mpDrawingStyle.get_default_hand_landmarks_style(),
                                        mpDrawingStyle.get_default_hand_connections_style())
            lms = get_landmarks(result)
            text = get_pred_from_landmarks(lms)
            if(old_text == text):
                count_same_frame += 1
            else:
                count_same_frame = 0
				
            if count_same_frame > SAME_FRAME_CNT:
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
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        keypress = cv2.waitKey(1)
        if keypress == ord('q') or keypress == ord('c'):
            break
        if keypress == ord('v') and is_voice_on:
            is_voice_on = False
        elif keypress == ord('v') and not is_voice_on:
            is_voice_on = True

    if keypress == ord('c'):
        return 2
    else:
        return 0

def recognize():
	cam = cv2.VideoCapture(1)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)
		else:
			break

recognize()
````

## Scores

```bash
    "loss": 0.0772174671292305,
    "accuracy": 0.983916163444519
```