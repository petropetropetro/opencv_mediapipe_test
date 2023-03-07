import cv2
import mediapipe as mp
import time
import numpy as np
import math
import HandTrackModule


def print_fingers(fingers):
    fingers_str = ['thumb', 'index  finger', 'middle finger', 'ring finger', 'pinky']
    output_str = ''
    num_of_fingers = 0
    if len(fingers) != 0:
        for i in range(0,5):
            if fingers[i] == 1:
                output_str = output_str + fingers_str[i] + " "
                num_of_fingers = num_of_fingers + 1
    return output_str, num_of_fingers


def main():

    detector = HandTrackModule.handDetector(detection_conf=0.7)

    cap = cv2.VideoCapture(0)
    # print(volume.GetVolumeRange()) (-65.25, 0.0, 0.75)   -65.25 is 0 sound volume and 0.0 is 100 volume in my case
    tip_ids = [4,8,12,16,20]
    pip_ids = [3,6,10,14,18]
    pTime = 0
  
    while True:

        success, img = cap.read()
        
        img = detector.findHands(img)
        LmList = detector.findPos(img)
        hand_type = detector.classifyHand()
        fingers = []
        if len(LmList) !=0:

            # LmList structure [id, x, y]
            # landmark 4 is thumb top point
            # landmark 8 is index finger top point
            # landmark 12 is middle finger top point
            # landmark 16 is ring finger top point
            # landmark 20 is pinky top point
            # idea to compare fingers top points with there pip landmarks coords (indx = 2,6,10,14,18)
            # specific condition to thumb because we can't compare it with y coords. so idea to compate it with x coord
            if hand_type[0] == 'Left':
                if LmList[tip_ids[0]][1] > LmList[pip_ids[0]][1]: #left hand
                    fingers.append(1)
                else:
                    fingers.append(0)
            elif hand_type[0] == 'Right':
                if LmList[tip_ids[0]][1] < LmList[pip_ids[0]][1]: #left hand
                    fingers.append(1)
                else:
                    fingers.append(0)

            for id in range(1,5):
                if LmList[tip_ids[id]][2] < LmList[pip_ids[id]][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        finger_str, num_of_fingers = print_fingers(fingers)
        output_finger_str = "Number of fingers: " + str(num_of_fingers) + " this fingers are opened: " + finger_str + "."
        cv2.putText(img, output_finger_str, (10,90), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 255), 1)
           
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break



if __name__ == "__main__":
    main()