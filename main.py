import os
from operator import length_hint

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

#variables
width, height = 1280, 720
folderPath = "Presentation"

#camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

#get the list of ppt images
pathImages = sorted(os.listdir(folderPath), key = len)
# print(pathImages)

#variables
imgNumber = 0
hs, ws = int(120*1), int(213*1)
gestureThreshold = 200
buttonPressed = False
buttonCounter = 0
buttonDelay = 40
annotations = [[]]
annotationNumber = -1
annotationStart = False

#Hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1) #detetctionCon tells if you are 80% sure that it's a hand, consider it as a hand

while True:
    #importing the images
    success, img = cap.read()
    img = cv2.flip(img, 1) #1 means flipping horizontically and 0 means flip vertically
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)


    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        #Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [640 // 2, 640], [0, width]))
        yVal = int(np.interp(lmList[8][1], [100, 480 - 100], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold: #if hand is at the height of the face
            #gesture 1 - left
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                annotationStart = False
                if imgNumber > 0:
                    buttonPressed = True
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False

            #gesture 2 - Right
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                annotationStart = False
                if imgNumber < len(pathImages)-1:
                    buttonPressed = True
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1

        #geature 3 - Show Pointer
        if fingers == [0, 1, 1, 0, 0]:
            annotationStart = False
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

        #gesture 4 - draw pointer
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)

        else:
            annotationStart = False


        #gesture 5 - erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                if annotationNumber >= 0:
                    annotations.pop(-1)
                    annotationNumber -= 1
                buttonPressed = True

    else:
        annotationStart = False

    #buttonPressed iterations
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False


    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

    #adding webcam image on the slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w-ws:w] = imgSmall

    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break