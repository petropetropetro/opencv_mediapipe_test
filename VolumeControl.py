import cv2
import mediapipe as mp
import time
import numpy as np
import math
import HandTrackModule


from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def main():

    detector = HandTrackModule.handDetector(detection_conf=0.7)
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    cap = cv2.VideoCapture(0)
    # print(volume.GetVolumeRange()) (-65.25, 0.0, 0.75)   -65.25 is 0 sound volume and 0.0 is 100 volume in my case
    volume_range = volume.GetVolumeRange()
    min_volume = volume_range[0]
    max_volume = volume_range[1]
    vol_bar_percent = np.interp(volume.GetMasterVolumeLevel(), [min_volume, max_volume], [0, 100])
    vol_bar = 400


    pTime = 0
  
    while True:

        success, img = cap.read()
        
        img = detector.findHands(img)
        LmList = detector.findPos(img)
        # based on mediapipe documentation
        # landmark 4 is thumb top point
        # landmark 8 is index finger top point

        if len(LmList) !=0:
            # LmList structure [id, x, y]
            x1, y1 = LmList[4][1], LmList[4][2]
            x2, y2 = LmList[8][1], LmList[8][2]

            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), 3)
            cv2.circle(img, (x1, y1), 5, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255,0,255), cv2.FILLED)

            line_len = math.hypot(x2-x1, y2-y1)
            #hand range [50; 300] 
            
            vol = np.interp(line_len, [50, 300], [min_volume, max_volume])
            vol_bar = np.interp(line_len, [50, 300], [400, 150])
            vol_bar_percent = np.interp(vol, [min_volume, max_volume], [0, 100])
            volume.SetMasterVolumeLevel(vol, None)

        # this value sometimes incorrect but its enough to demostrate the control possibility 
        cv2.rectangle(img, (50,150), (85, 400), (0,255,0),3)
        cv2.rectangle(img, (50,int(vol_bar)), (85, 400), (0,255,0), cv2.FILLED)
        cv2.putText(img, f'{str(int(vol_bar_percent))}%', (50,140), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

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