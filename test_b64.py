import base64
from deepface import DeepFace
import traceback

# valid jpeg base64
b64 = '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA='
img = 'data:image/jpeg;base64,' + b64
try:
    print("Testing DeepFace...")
    DeepFace.verify(img1_path=img, img2_path=img, enforce_detection=False)
    print("Success!")
except Exception as e:
    traceback.print_exc()
