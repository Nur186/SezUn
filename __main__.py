from __future__ import annotations

import logging
import os
import sys
import time

import cv2

from .config import CAMERA_INDEX, MODEL_NAME, SPEECH_COOLDOWN_SECONDS
from .detector import ObjectDetector
from .guidance import make_guidance
from .speech import Speaker


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        logging.error("Камера ашылмады. CAMERA_INDEX мәнін тексеріңіз.")
        return 1

    detector = ObjectDetector(MODEL_NAME)
    speaker = Speaker()
    last_spoken: dict[str, float] = {}
    logging.info("Көмекші іске қосылды. Тоқтату үшін терезеде q басыңыз.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                logging.warning("Камерадан кадр алынбады.")
                time.sleep(0.1)
                continue

            detections = detector.detect(frame)
            now = time.monotonic()
            messages: list[str] = []
            for detection in detections:
                message = make_guidance(detection, frame.shape[1])
                if message and now - last_spoken.get(message, 0) >= SPEECH_COOLDOWN_SECONDS:
                    messages.append(message)
                    last_spoken[message] = now

            # Keep audio brief: announce only the largest detection in this frame.
            if messages:
                speaker.say(messages[0])

            for detection in detections:
                x1, y1, x2, y2 = detection.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
                cv2.putText(
                    frame,
                    f"{detection.label} {detection.confidence:.0%}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 80),
                    2,
                )

            cv2.imshow("Assistive Vision — MVP", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        logging.info("Бағдарлама тоқтатылды.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        speaker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
