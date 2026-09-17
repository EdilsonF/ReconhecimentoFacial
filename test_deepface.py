import base64
import numpy as np
import cv2
from deepface import DeepFace
from deepface.commons import functions

# Try loading a dummy image
img = np.zeros((100, 100, 3), dtype=np.uint8)
try:
    img_loaded = functions.load_image(img)
    print("Loaded numpy array:", img_loaded[0].shape if isinstance(img_loaded, tuple) else img_loaded.shape)
except Exception as e:
    print("Error loading numpy array:", e)

try:
    result = DeepFace.verify(img1_path=img, img2_path=img, enforce_detection=False)
    print("Result:", result)
except Exception as e:
    print("Verify error:", type(e), e)
