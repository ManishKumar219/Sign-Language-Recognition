import asyncio
from typing import List, Tuple
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import cv2, pickle
import tensorflow as tf
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread
import numpy as np
import mediapipe as mp

# app =FastAPI() # initialize FastAPI
# initialize the classifier that we will use
model = load_model(r"artifacts/training/model.h5")
mpHands = mp.solutions.hands
hands = mpHands.Hands()


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

def get_operator(pred_text):
	try:
		pred_text = int(pred_text)
	except:
		return ""
	operator = ""
	if pred_text == 1:
		operator = "+"
	elif pred_text == 2:
		operator = "-"
	elif pred_text == 3:
		operator = "*"
	elif pred_text == 4:
		operator = "/"
	elif pred_text == 5:
		operator = "%"
	elif pred_text == 6:
		operator = "**"
	elif pred_text == 7:
		operator = ">>"
	elif pred_text == 8:
		operator = "<<"
	elif pred_text == 9:
		operator = "&"
	elif pred_text == 0:
		operator = "|"
	return operator

is_voice_on = True



@asynccontextmanager
async def lifespan(app: FastAPI):

    yield
    # print("Execute before closing")

app = FastAPI(lifespan=lifespan)


class Hands(BaseModel):

    hands_lms: List[str]

class Calculator(BaseModel):
    info: List[str]
    hands_lms: List[str]

async def receive(websocket: WebSocket, queue: asyncio.Queue):

    bytes = await websocket.receive_bytes()
    try:
        queue.put_nowait(bytes)
    except asyncio.QueueFull:
        pass

async def detect(websocket: WebSocket, queue: asyncio.Queue):

    text = ""
    word = ""
    count_same_frame = 0
    while True:
        bytes = await queue.get()
        data = np.frombuffer(bytes, dtype=np.uint8)
        img = cv2.imdecode(data, 1)
        frameRGB = process_image(img)
        result = hands.process(frameRGB)
        old_text = text
        if result.multi_hand_landmarks:
            lms = get_landmarks(result)
            text = get_pred_from_landmarks(lms)
            print(text)
            if(old_text == text):
                count_same_frame += 1
            else:
                count_same_frame = 0
                
            if count_same_frame > 10:
                word = word + text
                count_same_frame = 0
                print((word + "  ") * 50)
                hands_output = Hands(hands_lms=[word])
        elif(word!=""):
            text = ""
            if(old_text == text):
                count_same_frame += 1
            else:
                count_same_frame = 0

            if count_same_frame > 10:
                word = word + "<space>"
                count_same_frame = 0
                hands_output = Hands(hands_lms=[word])
        else:
            hands_output = Hands(hands_lms=[])

        await websocket.send_json(hands_output.dict())


# ----------------------------------------------------------------

# ----------------------------------------------------------------

async def calculatorMode(websocket: WebSocket, queue: asyncio.Queue):
    flag = {"first": False, "operator": False, "second": False, "clear": False}
    count_same_frames = 0
    first, operator, second = "", "", ""
    pred_text = ""
    calc_text = ""
    info = "Enter first number"
    SAME_FRAME_CNT = 6
    # Thread(target=say_text, args=(info,)).start()
    calc_output = Calculator(info=[info], hands_lms=[calc_text])
    count_clear_frames = 0

    while True:
        bytes = await queue.get()
        data = np.frombuffer(bytes, dtype=np.uint8)
        img = cv2.imdecode(data, 1)
        frameRGB = process_image(img)
        result = hands.process(frameRGB)
        old_text = pred_text
        if result.multi_hand_landmarks:
            lms = get_landmarks(result)
            pred_text = get_pred_from_landmarks(lms)

            if(old_text == pred_text):
                count_same_frames += 1
            else:
                count_same_frames = 0
             
            if (pred_text == "C" and count_same_frames > 10):
                count_same_frames = 0
                first, second, operator, pred_text, calc_text = '', '', '', '', ''
                flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
                info = "Enter first number"
                calc_output = Calculator(info=[info], hands_lms=[calc_text])
                # Thread(target=say_text, args=(info,)).start()

            elif pred_text == "Best of Luck " and count_same_frames > SAME_FRAME_CNT:
                count_same_frames = 0
                if flag['clear']:
                    first, second, operator, pred_text, calc_text = '', '', '', '', ''
                    flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
                    info = "Enter first number"
                    calc_output = Calculator(info=[info], hands_lms=[calc_text])
                    # Thread(target=say_text, args=(info,)).start()
                elif second != '':
                    flag['second'] = True
                    info = "Clear screen"
                    calc_output = Calculator(info=[info], hands_lms=[calc_text])
                    #Thread(target=say_text, args=(info,)).start()
                    second = ''
                    flag['clear'] = True
                    try:
                        calc_text += "= "+str(eval(calc_text))
                    except:
                        calc_text = "Invalid operation"
                    
                    calc_output = Calculator(info=["Result"], hands_lms=[calc_text])

                elif first != '':
                    flag['first'] = True
                    info = "Enter operator"
                    # Thread(target=say_text, args=(info,)).start()
                    first = ''
                    calc_output = Calculator(info=[info], hands_lms=[calc_text])

            elif pred_text != "Best of Luck " and pred_text.isnumeric():
                if flag['first'] == False:
                    if count_same_frames > SAME_FRAME_CNT:
                        count_same_frames = 0
                        # Thread(target=say_text, args=(pred_text,)).start()
                        first += pred_text
                        calc_text += pred_text
                        calc_output = Calculator(info=["Enter first number"], hands_lms=[calc_text])

                elif flag['operator'] == False:
                    operator = get_operator(pred_text)
                    if count_same_frames > SAME_FRAME_CNT:
                        count_same_frames = 0
                        flag['operator'] = True
                        calc_text += operator
                        info = "Enter second number"
                        calc_output = Calculator(info=[info], hands_lms=[calc_text])
                        # Thread(target=say_text, args=(info,)).start()
                        operator = ''
                elif flag['second'] == False:
                    if count_same_frames > SAME_FRAME_CNT:
                        # Thread(target=say_text, args=(pred_text,)).start()
                        second += pred_text
                        calc_text += pred_text
                        count_same_frames = 0	
                        calc_output = Calculator(info=["Enter second number"], hands_lms=[calc_text])

        if count_clear_frames == 30:
            first, second, operator, pred_text, calc_text = '', '', '', '', ''
            flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
            info = "Enter first number"
            # Thread(target=say_text, args=(info,)).start()
            count_clear_frames = 0
            calc_output = Calculator(info=[info], hands_lms=[])
    
        await websocket.send_json(calc_output.dict())


@app.websocket("/asl-detection")
async def hand_detection(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue(maxsize=10)
    detect_task = asyncio.create_task(detect(websocket, queue))
    try:
        while True:
            await receive(websocket, queue)
    except WebSocketDisconnect:
        detect_task.cancel()
        await websocket.close()


@app.websocket("/calc-mode")
async def hand_detection(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue(maxsize=10)
    detect_task = asyncio.create_task(calculatorMode(websocket, queue))
    try:
        while True:
            await receive(websocket, queue)
    except WebSocketDisconnect:
        detect_task.cancel()
        await websocket.close()
# @app.on_event("startup")
# async def startup():
