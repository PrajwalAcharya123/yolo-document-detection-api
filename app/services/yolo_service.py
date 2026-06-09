from ultralytics import YOLO
import os

model = None

MODEL_PATH = "best.pt"

if os.path.exists(MODEL_PATH):

    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

else:

    print("WARNING: best.pt not found. Running without model.")

