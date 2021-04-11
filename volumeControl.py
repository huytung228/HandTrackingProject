from HandDetectorModule import HandDetector
import mediapipe as mp
import numpy as np
import cv2
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# get volume range
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume.GetMute()
volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()
vol_max = vol_range[1]
vol_min = vol_range[0]

if __name__ == '__main__':
    # Calculate fps
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture('volctl.mp4')
    handDetectorObj = HandDetector(confidence=0.7)
    while(True):
        _, frame = cap.read()
        handDetectorObj.detect_hand(image=frame, draw=True)
        lm_cordinates = handDetectorObj.findPosition(frame, handNo=0)

        if len(lm_cordinates) != 0:
            point1 = lm_cordinates[4]
            point2 = lm_cordinates[8]
            cv2.circle(frame, point1, 6, (0, 0, 0), -1)
            cv2.circle(frame, point2, 6, (0, 0, 0), -1)
            # cv2.line(frame, point1, point2, (255, 0, 0), 4)
            length = math.hypot(point1[0] - point2[0], point1[1] - point2[1])
            # range 30 -220
            if length>220 or length<30:
                cv2.line(frame, point1, point2, (0, 0, 255), 4)
                if(length>220):
                    cv2.putText(frame, 'max vol', (370, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'min vol', (370, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.line(frame, point1, point2, (0, 153, 0), 4)

            # set volume level
            vol_current = np.interp(length, [30, 220], [vol_min, vol_max])
            volume.SetMasterVolumeLevel(vol_current, None)

            # draw percent bar
            volBar = np.interp(length, [30, 220], [400, 150])
            volPer = np.interp(length, [30, 220], [0, 100])
            cv2.rectangle(frame, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(frame, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)
                        
        # cal and show fps
        cTime = time.time()
        fps = str(int(1/(cTime - pTime)))
        pTime = cTime
        cv2.putText(frame, f'fps:{fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('paint', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break       

