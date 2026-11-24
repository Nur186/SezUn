import os

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
SPEECH_COOLDOWN_SECONDS = float(os.getenv("SPEECH_COOLDOWN_SECONDS", "3.0"))
