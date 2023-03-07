import cv2
import mediapipe as mp
import time
import math

class PoseEstimmation():
    def __init__(self, mode=False, model_complexity = 1, smooth = True, segmentation = False, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.segmentation = segmentation
        self.smooth = smooth
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth, self.segmentation,  self.detection_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def estimatePose(self, img, draw=False):
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks:
            if draw:
                 self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    

    def findPos(self, img, draw=False):
        
        self.lmList = []
        h,w,c = img.shape
        
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
        
        return  self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw = False):

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        #calculate angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

        if angle < 0:
            angle += 360

        if draw:
            cv2.circle(img, (x1, y1), 15, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255,0,255), cv2.FILLED) 
            cv2.circle(img, (x3, y3), 15, (255,0,255), cv2.FILLED)
        
        return angle
        


def main():

    cap = cv2.VideoCapture(0)
    detector = PoseEstimmation()
    pTime = 0
    cTime = 0
    
    while True:

        success, img = cap.read()
        
        img = detector.estimatePose(img, draw=True)
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