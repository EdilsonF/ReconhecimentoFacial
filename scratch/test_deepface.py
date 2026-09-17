import cv2
import numpy as np
from deepface import DeepFace

# create a dummy image (black square)
img = np.zeros((100, 100, 3), dtype=np.uint8)

try:
    res = DeepFace.verify(img1_path=img, img2_path=img, enforce_detection=True)
    print("Success")
except Exception as e:
    print(f"Error: {str(e)}")
