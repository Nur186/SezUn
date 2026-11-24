from .detector import Detection

KAZAKH_LABELS = {
    "person": "адам",
    "bicycle": "велосипед",
    "car": "көлік",
    "motorcycle": "мотоцикл",
    "bus": "автобус",
    "truck": "жүк көлігі",
    "traffic light": "бағдаршам",
    "stop sign": "тоқта белгісі",
    "chair": "орындық",
    "bench": "орындық",
    "dog": "ит",
    "stairs": "баспалдақ",
}


def make_guidance(detection: Detection, frame_width: int) -> str:
    """Turn a detection into a short directional cue (not a safety guarantee)."""
    x1, _, x2, y2 = detection.box
    center_x = (x1 + x2) / 2
    position = "сол жақта" if center_x < frame_width / 3 else "оң жақта" if center_x > frame_width * 2 / 3 else "алдыңызда"
    # Use the bounding box's frame-area ratio as a rough prominence cue, not meters.
    frame_height = max(y2, 1)
    box_area_ratio = max(0.0, (x2 - x1) * (y2 - detection.box[1]) / (frame_width * frame_height))
    label = KAZAKH_LABELS.get(detection.label, detection.label)
    prominence = " Ірі көрінеді." if box_area_ratio > 0.18 else ""
    return f"{label}, {position}.{prominence}"
