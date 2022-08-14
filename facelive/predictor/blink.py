from dataclasses import dataclass
from typing import Counter
import imutils
import numpy as np
from scipy.spatial import distance as dist
from facelive.predictor.landmark import LandmarkPredictor


@dataclass
class Eyes:
    left:int = 0
    right:int = 0
    both:int = 0
    
@dataclass
class CounterTotal:
    counter: Eyes = Eyes()
    total: Eyes = Eyes()
    
@dataclass
class EAR:
    left:float = 0
    right:float = 0
    both:float = 0
    

class BlinkDetector:
    def __init__(self, left_thres=0.16, right_thres=0.16, eyes_thres=0.26, consec_frame=3):
        self.left_thres = left_thres
        self.right_thres = right_thres
        self.eyes_thres = eyes_thres
        self.consec_frame = consec_frame
        
        self.data = CounterTotal()
        

    def _both_blink(self, rblink, lblink):
        if rblink and lblink:
            self.data.counter.both += 1
        else:
            if self.data.counter.both>=self.consec_frame:
                self.data.total.both +=1
            self.data.counter.both = 0
                
    def _left_blink(self, rblink, lblink):
        if lblink and not rblink:
            self.data.counter.left += 1
        else:
            if self.data.counter.left>=self.consec_frame:
                self.data.total.left +=1
            self.data.counter.left = 0
            
    def _right_blink(self, rblink, lblink):
        if rblink and not lblink:
            self.data.counter.right += 1
        else:
            if self.data.counter.right>=self.consec_frame:
                self.data.total.right +=1
            self.data.counter.right = 0
        
    def _blink_counter(self, rblink, lblink):
        self._both_blink(rblink, lblink)
        self._right_blink(rblink, lblink)
        self._left_blink(rblink, lblink)
        
    def _blink_status(self, left_ear, right_ear, both_ear):
        ls, rs, es = False, False, False 
        if left_ear < self.left_thres: ls = True
        if right_ear < self.right_thres: rs = True
        if both_ear < self.eyes_thres: es = True
        
        return rs, ls, es
        
    def detect(self, landmark:np.ndarray)->dict:
        right_ear, left_ear, both_ear = self.eyes_aspect_ratio(landmark)
        rstat, lstat, estat = self._blink_status(right_ear, left_ear, both_ear)
        self._blink_counter(rstat, lstat)
        
        return {
            "ear": EAR(left=left_ear, right=right_ear, both=both_ear),
            "blink": self.data.total
        }

    def eyes_landmark(self, landmark:np.array)->tuple:
        ls,le = imutils.face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        rs,re = imutils.face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        left_eye, right_eye = landmark[ls:le], landmark[rs:re]
        
        return left_eye, right_eye
    
    def eye_aspect_ratio(self, eye)->float:
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        
        ear = (A + B) / (2.0 * C)

        return ear

    def eyes_aspect_ratio(self, landmark:np.array)->tuple:
        left_lm, right_lm = self.eyes_landmark(landmark)
        left_ear = self.eye_aspect_ratio(left_lm)
        right_ear = self.eye_aspect_ratio(right_lm)
        eyes_ear = (left_ear + right_ear) / 2
        return right_ear, left_ear, eyes_ear



