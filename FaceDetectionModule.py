import cv2
import mediapipe as mp

import time



class FaceDetection():
    def __init__(self, selfie_mode = False, detection_conf=0.5):
        self.selfie_mode = selfie_mode
        self.detection_conf = detection_conf
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.selfie_mode, self.detection_conf)

    def detection(self, img):
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,c = img.shape
        self.results = self.face_detection.process(img_rgb)
        
        Lbbox = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin *w), int(bboxC.ymin *h), int(bboxC.width *w), int(bboxC.height *h)
                Lbbox.append([id, bbox, int(detection.score[0] * 100)])

        return img, Lbbox
    
    def draw(self, img, list_bbox):
        if len(list_bbox) > 0:
            cv2.rectangle(img, list_bbox[0][1], (255,0,255),2)
            cv2.putText(img, f'{str(list_bbox[0][2])}%', (list_bbox[0][1][0],list_bbox[0][1][1] - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)


def main():

    cap = cv2.VideoCapture(0)
    detector = FaceDetection()
    pTime = 0
    cTime = 0
    
    while True:

        success, img = cap.read()
        list_bbox = []
        img, list_bbox = detector.detection(img)
        detector.draw(img, list_bbox)
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