import cv2
import mediapipe as mp
import time
import numpy as np

class HandDetector():
    def __init__(self, mode=False, max_hands = 2, confidence=0.5):
        # Create hands object
        self.mpHands = mp.solutions.hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=confidence) 

        # Drawing ultils
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_hand(self, image, draw=True):
        self.results = self.mpHands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not self.results.multi_hand_landmarks:
            print("There no hand in image")
            return
        if draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    def findPosition(self, image, handNo, draw=True):
        h , w = image.shape[:2]
        lm_cordinates = []
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_x, lm_y = int(lm.x * w), int(lm.y * h)
                lm_cordinates.append((lm_x, lm_y))
        return lm_cordinates

if __name__ == '__main__':
    # Calculate fps
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture('1hand.mp4')
    handDetectorObj = HandDetector(confidence=0.7)
    imgResult = None
    x1,y1=0,0
    while(True):
        _, frame = cap.read()
        if imgResult is None:
            imgResult = np.zeros_like(frame)
        handDetectorObj.detect_hand(image=frame, draw=False)
        lm_cordinates = handDetectorObj.findPosition(frame, handNo=0)
        if len(lm_cordinates) != 0:
            (x2,y2) = lm_cordinates[4]
            # cv2.circle(frame, lm_cordinates[4], 10, (255, 255, 0), -1)
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
            else:
                # Draw the line on the canvas
                imgResult = cv2.line(imgResult, (x1,y1),(x2,y2), [255,255,0], 4)
            # After the line is drawn the new points become the previous points.
            x1,y1= x2,y2
        else:
            x1,y1=(0,0)
        frame = cv2.add(frame,imgResult)
        stacked = np.hstack((imgResult,frame))
        cv2.imshow('paint', stacked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break       




    