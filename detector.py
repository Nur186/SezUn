from dataclasses import dataclass

from ultralytics import YOLO

from .config import CONFIDENCE_THRESHOLD


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class ObjectDetector:
    def __init__(self, model_name: str) -> None:
        self.model = YOLO(model_name)

    def detect(self, frame) -> list[Detection]:
        result = self.model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        found: list[Detection] = []
        for item in result.boxes:
            class_id = int(item.cls[0].item())
            label = str(self.model.names[class_id])
            confidence = float(item.conf[0].item())
            x1, y1, x2, y2 = (int(value) for value in item.xyxy[0].tolist())
            found.append(Detection(label, confidence, (x1, y1, x2, y2)))
        # Largest image footprint first; this is not a calibrated distance estimate.
        found.sort(key=lambda item: (item.box[2] - item.box[0]) * (item.box[3] - item.box[1]), reverse=True)
        return found
