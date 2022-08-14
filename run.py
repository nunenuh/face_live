# import the necessary package

import argparse
from facelive.runner import DetectorRunner

if __name__ == "__main__":
    
    detector = DetectorRunner(
        landmark_weight='./weights/pipnet/pipnet_resnet18_300w.onnx',
        emotion_weight='./weights/env2s_fer.onnx',
    )
    
    detector.run()
  