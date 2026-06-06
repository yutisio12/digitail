import io
import numpy as np
import cv2
import easyocr

MODEL_VERSION = "easyocr-1.0"

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(lang=["en"], gpu=False)
    return _reader


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = thresh.shape
    if h < 300:
        scale = 300 / h
        new_w = int(w * scale)
        thresh = cv2.resize(thresh, (new_w, 300))

    return thresh


def read_meter(image_bytes: bytes, meter_type: str) -> dict:
    processed = preprocess_image(image_bytes)

    reader = _get_reader()
    results = reader.readtext(processed)

    allowed_chars = set("0123456789.")
    filtered = []
    for bbox, text, conf in results:
        cleaned = "".join(ch for ch in text if ch in allowed_chars)
        if cleaned:
            filtered.append((bbox, cleaned, conf))

    if not filtered:
        return {
            "reading": None,
            "confidence": 0.0,
            "bounding_boxes": [],
            "model_version": MODEL_VERSION,
        }

    reading = "".join(t[1] for t in filtered)
    avg_conf = float(np.mean([t[2] for t in filtered]))
    boxes = [
        {"points": [[float(p[0]), float(p[1])] for p in t[0]], "text": t[1], "confidence": float(t[2])}
        for t in filtered
    ]

    return {
        "reading": reading,
        "confidence": round(avg_conf, 4),
        "bounding_boxes": boxes,
        "model_version": MODEL_VERSION,
    }
