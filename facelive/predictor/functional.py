
from PIL import Image
import numpy as np

def crop_face(image, boxes, pad=0.1):
    xmin,ymin,xmax,ymax, conf = boxes
    w,h = xmax-xmin, ymax-ymin
    xmin = xmin - int(w*pad)
    ymin = ymin - int(h*pad)
    xmax = xmax + int(w*pad)
    ymax = ymax + int(h*pad)
    face = image[ymin:ymax, xmin:xmax]
    
    return face

def convert_cv_to_pil_grayscale(image):
    img = Image.fromarray(image)
    img = img.convert("L")
    
    return img

def face_emotion_crop(image, boxes):
    face = crop_face(image, boxes)
    face = convert_cv_to_pil_grayscale(face)
    return face