import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self, mode=False, max_numb_hands=2, modeIC=1, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.modeIC = modeIC
        self.max_numb_hands = max_numb_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_numb_hands, self.modeIC, self.detection_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=False):
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    

    def findPos(self, img, handNo=0, draw=False):
        
        lmList = []
        h,w,c = img.shape
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
        
        return lmList
    
    def classifyHand(self):

        handsType = []
        if self.results.multi_hand_landmarks:
            for hand in  self.results.multi_handedness:
                handType=hand.classification[0].label
                handsType.append(handType)
        return handsType
        


def main():

    cap = cv2.VideoCapture(0)
    detector = handDetector()
    pTime = 0
    cTime = 0
    
    while True:

        success, img = cap.read()
        
        img = detector.findHands(img, draw=True)
        lmList = detector.findPos(img, draw=True)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break



if __name__ == "__main__":
    main()