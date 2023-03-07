import cv2
import mediapipe as mp
import time
import PoseEstmModule
import numpy as np

def main():

    cap = cv2.VideoCapture(0)
    detector = PoseEstmModule.PoseEstimmation()
    pTime = 0
    cTime = 0
    count = 0
    dir = 0
    while True:
        
        success, img = cap.read()
        
        img = detector.estimatePose(img, draw=False)
        lmList = detector.findPos(img, draw=False)

        # landmark 12,14,16 are arm points elbow, shoulder and wrist
        if len(lmList) != 0:
            angle = detector.findAngle(img, 12, 14, 16, draw=True)
            per = np.interp(angle,(210,300),(0,100))

            if per == 100:
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                if dir == 1:
                    count += 0.5
                    dir = 0



        cv2.putText(img, f'Num of done iteration {str(count)}', (10,90), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)

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