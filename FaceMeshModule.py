import cv2
import mediapipe as mp

import time



class FaceMesh():
    def __init__(self, mode = False, max_num_of_face = 2, refine_landmarks = False, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_num_of_face = max_num_of_face
        self.refine_landmarks = refine_landmarks
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.mode, self.max_num_of_face, self.refine_landmarks, self.detection_conf, self.track_conf )

    def detection(self, img):
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,c = img.shape
        self.results = self.face_mesh.process(img_rgb)
        
        faces = []

        if self.results.multi_face_landmarks:
            for faceLm in self.results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(img, faceLm, self.mp_face_mesh.FACEMESH_TESSELATION)
                face = []
                for id, lm in enumerate(faceLm.landmark):
                    x, y = int(lm.x*w), int(lm.y*h)
                    face.append([id, x, y])
                faces.append([faceLm, face])

        return img, faces
    
def main():

    cap = cv2.VideoCapture(0)
    detector = FaceMesh()
    pTime = 0
    cTime = 0
    
    while True:

        success, img = cap.read()
        faces_list = []
        img, faces_list = detector.detection(img)

        if faces_list:
            print(len(faces_list))
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