try:  # Support both package execution and direct local development.
    from .detector import Detection
except ImportError:
    from detector import Detection

KAZAKH_LABELS = {
    "person": "адам",
    "bicycle": "велосипед",
    "car": "көлік",
    "motorcycle": "мотоцикл",
    "airplane": "ұшақ",
    "bus": "автобус",
    "train": "пойыз",
    "truck": "жүк көлігі",
    "boat": "қайық",
    "traffic light": "бағдаршам",
    "fire hydrant": "өрт гидранты",
    "stop sign": "тоқта белгісі",
    "parking meter": "тұрақ есептегіші",
    "bench": "орындық",
    "bird": "құс",
    "cat": "мысық",
    "dog": "ит",
    "horse": "жылқы",
    "sheep": "қой",
    "cow": "сиыр",
    "elephant": "піл",
    "bear": "аю",
    "zebra": "зебра",
    "giraffe": "керік",
    "backpack": "арқа сөмкесі",
    "umbrella": "қолшатыр",
    "handbag": "сөмке",
    "tie": "галстук",
    "suitcase": "шабадан",
    "frisbee": "фрисби",
    "skis": "шаңғы",
    "snowboard": "сноуборд",
    "sports ball": "доп",
    "kite": "батпырауық",
    "baseball bat": "бейсбол таяғы",
    "baseball glove": "бейсбол қолғабы",
    "skateboard": "скейтборд",
    "surfboard": "серфинг тақтасы",
    "tennis racket": "теннис ракеткасы",
    "bottle": "бөтелке",
    "wine glass": "бокал",
    "cup": "кесе",
    "fork": "шанышқы",
    "knife": "пышақ",
    "spoon": "қасық",
    "bowl": "тостаған",
    "banana": "банан",
    "apple": "алма",
    "sandwich": "сэндвич",
    "orange": "апельсин",
    "broccoli": "брокколи",
    "carrot": "сәбіз",
    "hot dog": "хот-дог",
    "pizza": "пицца",
    "donut": "пончик",
    "cake": "торт",
    "chair": "орындық",
    "couch": "диван",
    "potted plant": "құмырадағы өсімдік",
    "bed": "төсек",
    "dining table": "ас үстелі",
    "toilet": "дәретхана",
    "tv": "теледидар",
    "laptop": "ноутбук",
    "mouse": "компьютер тінтуірі",
    "remote": "қашықтан басқару пульті",
    "keyboard": "пернетақта",
    "cell phone": "ұялы телефон",
    "microwave": "микротолқынды пеш",
    "oven": "пеш",
    "toaster": "тостер",
    "sink": "қолжуғыш",
    "refrigerator": "тоңазытқыш",
    "book": "кітап",
    "clock": "сағат",
    "vase": "құмыра",
    "scissors": "қайшы",
    "teddy bear": "ойыншық аю",
    "hair drier": "фен",
    "toothbrush": "тіс щеткасы",
    "stairs": "баспалдақ",
}


def kazakh_label(label: str) -> str:
    """Return a Kazakh name for a detector class, including unknown classes."""
    return KAZAKH_LABELS.get(label, "анықталмаған нысан")


def make_guidance(detection: Detection, frame_width: int) -> str:
    """Turn a detection into a short directional cue (not a safety guarantee)."""
    x1, _, x2, y2 = detection.box
    center_x = (x1 + x2) / 2
    position = "сол жақта" if center_x < frame_width / 3 else "оң жақта" if center_x > frame_width * 2 / 3 else "алдыңызда"
    # Use the bounding box's frame-area ratio as a rough prominence cue, not meters.
    frame_height = max(y2, 1)
    box_area_ratio = max(0.0, (x2 - x1) * (y2 - detection.box[1]) / (frame_width * frame_height))
    label = kazakh_label(detection.label)
    prominence = " Ірі көрінеді." if box_area_ratio > 0.18 else ""
    return f"{label}, {position}.{prominence}"
