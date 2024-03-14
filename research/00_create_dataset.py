import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import sqlite3
import os
import tensorflow as tf
from IPython.display import display
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

landmarks = []
lables = []

# Connect to the Camera
cam = cv2.VideoCapture(0)

# Connect to the Dataset
conn = sqlite3.connect('lables.db')
cursor = conn.cursor()
command = "SELECT * FROM gesture"
cursor.execute(command)
classes = []
for row in cursor:
    classes.append(row[1])
print(classes)
    
# Create Mediapipe objects
mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils
mpDrawingStyle = mp.solutions.drawing_styles

# Columns of the Dataset
cols = []
for i in range(21):\
    cols.extend([f'{i}x', f'{i}y', f'{i}z'])
# cols.append('label')


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
        # lmsList = [lmsList]
        # lmsList = np.array(lmsList)
    return lmsList




while True:
        img = cam.read()[1]
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
        cv2.putText(blackboard, "Press 'S' to START", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 0))

        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

NO_OF_SAMPLES = 10000
i = 0
for cls in classes:
    cls_dataset = []
    cls_lable = []
    cnt = NO_OF_SAMPLES
    print(cls)
    i+=1
        
    while cnt:
        img = cam.read()[1]
        imgRGB = process_image(img)
        result = hands.process(imgRGB)

        if result.multi_hand_landmarks:
            handLms = result.multi_hand_landmarks[0]
            lms = get_landmarks(result)
            # print([lms])
            landmarks.append(lms)
            cls_dataset.append(lms)
            lables.append(cls)
            cls_lable.append(cls)
            cnt -= 1
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, mpDrawingStyle.get_default_hand_landmarks_style(),
                                    mpDrawingStyle.get_default_hand_connections_style())
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
        cv2.putText(blackboard, "Current Class- ", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 0))
        cv2.putText(blackboard, cls, (30, 160), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 0))
        cv2.putText(blackboard, "Count- " + str(NO_OF_SAMPLES-cnt), (30, 230), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 0))


        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    try:
        np_landmarks = np.array(cls_dataset)
        df = pd.DataFrame(np_landmarks, columns=cols)
        df['lable'] = cls_lable
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv(f'research/Data/{cls}_gesture_wlms2.csv', index=False)
    except:
        print(f"Failed to Read Samples of {cls}")
        pass

    while True:
        img = cam.read()[1]

        imgRGB = process_image(img)
        result = hands.process(imgRGB)

        if result.multi_hand_landmarks:
            handLms = result.multi_hand_landmarks[0]
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, mpDrawingStyle.get_default_hand_landmarks_style(),
                                        mpDrawingStyle.get_default_hand_connections_style())
            
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
        cv2.putText(blackboard, "Press N for Next Class", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 0))
        try:
            cv2.putText(blackboard, "Next class: " + classes[i], (30, 180), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 0))
        except:
            pass
        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        
        if cv2.waitKey(1) & 0xFF == ord('n'):
            break

print("\n\n=====================\n\nLandmarks are: ",landmarks)

print("Shape of all landmarks",np.array(landmarks).shape)
print("Lenth of classes", len(classes))
print("Expected shape of Landmarks = ", len(classes)* NO_OF_SAMPLES)

# np_landmarks = np.array(landmarks)

# df = pd.DataFrame(np_landmarks, columns=cols)
# df['lable'] = lables

# df = df.sample(frac=1).reset_index(drop=True)

# df.to_csv('research/Data/Data.csv', index=False)

# display(df.head())


############# Code for Merging all the datasets #############

# BASE_PATH = "research/Data/"
# list_dir = os.listdir("research\Data")

# df = pd.DataFrame()

# for i in list_dir:
#     temp = pd.read_csv(BASE_PATH + i)
#     df = pd.concat([df, temp])
# print(df.shape)

# df = df.sample(frac=1).reset_index(drop=True)

# df.to_csv(f'research/Data/Data.csv', index=False)
# df.head()