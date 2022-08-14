import cv2 
import time
import imutils
from facelive.predictor.landmark import LandmarkPredictor
from facelive.predictor.emotion import EmotionDetector
from facelive.predictor.blink import BlinkDetector, Eyes, EAR
import facelive.predictor.functional as F
from imutils import face_utils

import numpy as np


class DetectorRunner:
    def __init__(self, landmark_weight:str, emotion_weight:str, cam=0):
        self.cam = cam    
        self.landmark_weight = landmark_weight
        self.emotion_weight = emotion_weight
            
        self._init_detector()
        self._init_attr()
        self._initialize_camera()
    
    def _init_attr(self):
        self.face_exp = "Not Smile"
        self.blink = Eyes()
        self.ear = EAR()
        
    def _init_detector(self):
        self.landmarknet = LandmarkPredictor(self.landmark_weight)
        self.emodetnet = EmotionDetector(weight=self.emotion_weight, topk=1)
        self.blinkdet = BlinkDetector(left_thres=0.16, right_thres=0.16, eyes_thres=0.20, consec_frame=2)
        
    def _initialize_camera(self):
        self.vs = cv2.VideoCapture(self.cam)
        if not self.vs.isOpened():
            print("Cannot open camera")
            exit()
        time.sleep(1.0)
        
    def _build_frame(self, src_frame, width=800):
        src_frame = imutils.resize(src_frame, width=width)
        src_frame = cv2.flip(src_frame,1)
        base_frame = imutils.resize(src_frame, width=width)
        top_bar = (np.ones((87,width, 3)) * 137).astype(np.uint8)
        bottom_bar = (np.ones((54+13,width, 3)) * 137).astype(np.uint8)
        show_frame = cv2.vconcat([top_bar, base_frame, bottom_bar])
        
        return show_frame
        
    def _emotion_detection(self, frame, boxes):
        face_img = F.face_emotion_crop(frame, boxes)
        emo = self.emodetnet.predict(face_img)
        
        if emo["class"]=="happy":
            self.face_exp = "Smile"
        else:
            self.face_exp = "Not Smile"
        
    def _blink_detection(self, landmark):
        data = self.blinkdet.detect(landmark)
        self.blink = data["blink"]
        self.ear = data['ear']
        
    def _detection(self, frame):
        frame, landmarks, bboxes = self.landmarknet.predict(frame, draw_boxes=True)
        if len(bboxes)>0:
            boxes, landmark = bboxes[0].astype(np.int32), landmarks[0]
            self._blink_detection(landmark)
            self._emotion_detection(frame, boxes)
        return frame
        
    def _draw_info(self, show_frame):
        cv2.putText(show_frame, "Left Blink", (50, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.putText(show_frame, f"{self.blink.left:03d}", (85, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        

        cv2.putText(show_frame, "Both Blink", (345, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(show_frame, f"{self.blink.both:03d}", (380, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        
        cv2.putText(show_frame, "Right Blink", (635, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(show_frame, f"{self.blink.right:03d}", (670, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        

        cv2.putText(show_frame, "Facial Expression", (320, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        if self.face_exp=="Smile":
            cv2.putText(show_frame, "Smile", (380, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 64, 255), 2)
        else:
            cv2.putText(show_frame, "Not Smile", (355, 740), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        
    def run(self):
        while True:

            ret, frame = self.vs.read()
            frame = self._build_frame(frame)
            
            try:
                frame = self._detection(frame)
            except Exception as e:
                print(e)
                
            self._draw_info(frame)
            
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
        
            if key == ord("q"):
                break;
            
        cv2.destroyAllWindows() 
            
        

        
    


