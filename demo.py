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
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
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
# def calculator_mode(cam):
#      pass
    
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

def calculator_mode(cam):
    global is_voice_on
    flag = {"first": False, "operator": False, "second": False, "clear": False}
    count_same_frames = 0
    first, operator, second = "", "", ""
    pred_text = ""
    calc_text = ""
    info = "Enter first number"
    Thread(target=say_text, args=(info,)).start()
    count_clear_frames = 0

    while True:
        img = cam.read()[1]
        imgRGB = process_image(img)
        result = hands.process(imgRGB)
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
                Thread(target=say_text, args=(info,)).start()

            elif pred_text == "Best of Luck " and count_same_frames > SAME_FRAME_CNT:
                count_same_frames = 0
                if flag['clear']:
                    first, second, operator, pred_text, calc_text = '', '', '', '', ''
                    flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
                    info = "Enter first number"
                    Thread(target=say_text, args=(info,)).start()
                elif second != '':
                    flag['second'] = True
                    info = "Clear screen"
                    #Thread(target=say_text, args=(info,)).start()
                    second = ''
                    flag['clear'] = True
                    try:
                        calc_text += "= "+str(eval(calc_text))
                    except:
                        calc_text = "Invalid operation"
                    if is_voice_on:
                        speech = calc_text
                        speech = speech.replace('-', ' minus ')
                        speech = speech.replace('/', ' divided by ')
                        speech = speech.replace('**', ' raised to the power ')
                        speech = speech.replace('*', ' multiplied by ')
                        speech = speech.replace('%', ' mod ')
                        speech = speech.replace('>>', ' bitwise right shift ')
                        speech = speech.replace('<<', ' bitwise leftt shift ')
                        speech = speech.replace('&', ' bitwise and ')
                        speech = speech.replace('|', ' bitwise or ')
                        Thread(target=say_text, args=(speech,)).start()
                elif first != '':
                    flag['first'] = True
                    info = "Enter operator"
                    Thread(target=say_text, args=(info,)).start()
                    first = ''

            elif pred_text != "Best of Luck " and pred_text.isnumeric():
                if flag['first'] == False:
                    if count_same_frames > SAME_FRAME_CNT:
                        count_same_frames = 0
                        Thread(target=say_text, args=(pred_text,)).start()
                        first += pred_text
                        calc_text += pred_text
                elif flag['operator'] == False:
                    operator = get_operator(pred_text)
                    if count_same_frames > SAME_FRAME_CNT:
                        count_same_frames = 0
                        flag['operator'] = True
                        calc_text += operator
                        info = "Enter second number"
                        Thread(target=say_text, args=(info,)).start()
                        operator = ''
                elif flag['second'] == False:
                    if count_same_frames > SAME_FRAME_CNT:
                        Thread(target=say_text, args=(pred_text,)).start()
                        second += pred_text
                        calc_text += pred_text
                        count_same_frames = 0	

        if count_clear_frames == 30:
            first, second, operator, pred_text, calc_text = '', '', '', '', ''
            flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
            info = "Enter first number"
            Thread(target=say_text, args=(info,)).start()
            count_clear_frames = 0

        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Calculator Mode", (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
        cv2.putText(blackboard, "Predicted text- " + pred_text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, "Operator " + operator, (30, 140), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
        cv2.putText(blackboard, calc_text, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        cv2.putText(blackboard, info, (30, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255) )
        if is_voice_on:
            cv2.putText(blackboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
        else:
            cv2.putText(blackboard, " ", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        keypress = cv2.waitKey(1)
        if keypress == ord('q') or keypress == ord('t'):
            break
        if keypress == ord('v') and is_voice_on:
            is_voice_on = False
        elif keypress == ord('v') and not is_voice_on:
            is_voice_on = True

    if keypress == ord('t'):
        return 1
    else:
        return 0

def recognize():
	cam = cv2.VideoCapture(0)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)
		elif keypress == 2:
			keypress = calculator_mode(cam)
		else:
			break

recognize()