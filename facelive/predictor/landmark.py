import torchlm
from torchlm.runtime import faceboxesv2_ort, pipnet_ort

class LandmarkPredictor:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self._build()
    
    def _build(self):
        torchlm.runtime.bind(faceboxesv2_ort())
        torchlm.runtime.bind(
            pipnet_ort(onnx_path=self.onnx_path, num_nb=10,
                       num_lms=68, net_stride=32, input_size=256, 
                       meanface_type="300w")
        )
    
    def predict(self, image, draw_landmark=False, draw_boxes=False):
        landmarks, bboxes = torchlm.runtime.forward(image)
        if draw_landmark:
            image = torchlm.utils.draw_landmarks(image, landmarks=landmarks)
        if draw_boxes:
            image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
        
        return image, landmarks, bboxes
        