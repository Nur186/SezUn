"""Local web interface for the camera based assistive vision prototype."""
from __future__ import annotations

import base64
import logging
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    from .config import MODEL_NAME
    from .detector import ObjectDetector
    from .guidance import kazakh_label, make_guidance
except ImportError:  # python app.py from the project folder
    from config import MODEL_NAME
    from detector import ObjectDetector
    from guidance import kazakh_label, make_guidance

app = Flask(__name__)
detector: ObjectDetector | None = None


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/detect")
def detect():
    global detector
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image", "")
    if not image_data.startswith("data:image/") or "," not in image_data:
        return jsonify(error="Камерадан кадр алынбады."), 400
    try:
        raw = base64.b64decode(image_data.split(",", 1)[1], validate=True)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image")
        if detector is None:
            detector = ObjectDetector(MODEL_NAME)
        found = detector.detect(frame)
        return jsonify(objects=[{
            "label": kazakh_label(item.label),
            "confidence": round(item.confidence, 2),
            "box": list(item.box),
            "message": make_guidance(item, frame.shape[1]),
        } for item in found[:5]])
    except Exception:
        logging.exception("Кадрды өңдеу сәтсіз аяқталды")
        return jsonify(error="Кадрды өңдеу мүмкін болмады. Сервер журналын тексеріңіз."), 500


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
