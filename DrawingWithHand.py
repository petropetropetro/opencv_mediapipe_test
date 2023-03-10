import cv2
import mediapipe as mp
import time
import numpy as np
import math
import HandTrackModule


def main():

    detector = HandTrackModule.handDetector(detection_conf=0.7)
    cap = cv2.VideoCapture(0)
    xp, yp =  0, 0
    imgCanvas = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), np.uint8)
    
    pTime = 0
  
    while True:

        success, img = cap.read()
        
        img = detector.findHands(img)
        LmList = detector.findPos(img)

        if len(LmList) !=0:
            # LmList structure [id, x, y]
            # landmark 4 is thumb top point
            # landmark 8 is index finger top point
            # landmark 12 is middle finger top point
            # drawing by two fingers thumb+index finger 
            # cleaning by two fingers index+middle 
            x1, y1 = LmList[4][1], LmList[4][2]
            x2, y2 = LmList[8][1], LmList[8][2]
            x3, y3 = LmList[12][1], LmList[12][2]

            draw_line_len = math.hypot(x2-x1, y2-y1)
            clear_line_len = math.hypot(x3-x2, y3-y2)

            if draw_line_len < 50:
                if xp == 0 and yp == 0:
                    xp, yp = (x1+x2)//2, (y1+y2)//2
                cv2.line(imgCanvas, (xp,yp), (xp,yp), (255,0,255), 7)
                xp, yp = (x1+x2)//2, (y1+y2)//2

            if clear_line_len < 50:
                if xp == 0 and yp == 0:
                    xp, yp = (x2+x3)//2, (y2+y3)//2
                cv2.line(imgCanvas, (xp,yp), (xp,yp), (0,0,0), 15)
                xp, yp = (x2+x3)//2, (y2+y3)//2

        imgCanvasGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(imgCanvasGray, 50, 255, cv2.THRESH_BINARY_INV)
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, thresholded)
        img = cv2.bitwise_or(img, imgCanvas)
        # or use to draw   
        # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)
        cv2.imshow("Image", img)
        cv2.imshow("Canvas", imgCanvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break



if __name__ == "__main__":
    main()