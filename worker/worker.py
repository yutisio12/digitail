import os
import time
import json
import cv2
import numpy as np
import onnxruntime as ort

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/best3.onnx"
QUEUE_DIR = "queue"
RESULT_DIR = "results"

IMG_SIZE = 640
CONF_THRES = 0.15
IOU_THRES = 0.5

os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# ONNX SESSION
# =========================
session = ort.InferenceSession(
    MODEL_PATH,
    providers=ort.get_available_providers()
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("ONNX Providers:", session.get_providers())
print("Input:", input_name)
print("Output:", output_name)

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1] / 255.0  # BGR → RGB
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32)
    return img

# =========================
# IOU
# =========================
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

# =========================
# PARSE YOLOv8 ONNX OUTPUT
# =========================
def parse_digits(outputs):
    preds = outputs[0]

    print("RAW OUTPUT SHAPE:", preds.shape)
    print("RAW OUTPUT SAMPLE:", preds.reshape(-1)[:10])

    # (1, 14, 8400) → (8400, 14)
    preds = preds[0].T

    candidates = []

    for row in preds:
        x, y, w, h = row[:4]
        obj_logit = row[4]
        class_logits = row[5:]

        obj_conf = sigmoid(obj_logit)
        class_scores = sigmoid(class_logits)

        cls_id = int(np.argmax(class_scores))
        cls_conf = class_scores[cls_id]

        conf = obj_conf * cls_conf

        if conf < CONF_THRES:
            continue

        # NOTE: box sudah pixel-based (BUKAN normalized)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        candidates.append({
            "box": [x1, y1, x2, y2],
            "digit": cls_id,
            "conf": float(conf)
        })

    if not candidates:
        return None, 0.0

    # NMS
    candidates.sort(key=lambda c: c["conf"], reverse=True)
    final = []

    while candidates:
        best = candidates.pop(0)
        final.append(best)
        candidates = [
            c for c in candidates
            if iou(best["box"], c["box"]) < IOU_THRES
        ]

    final.sort(key=lambda b: b["box"][0])

    number = "".join(str(b["digit"]) for b in final)
    avg_conf = float(np.mean([b["conf"] for b in final]))

    return number, avg_conf

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================
# WORKER LOOP
# =========================
def worker_loop():
    print("Worker started, waiting for jobs...")

    while True:
        files = [f for f in os.listdir(QUEUE_DIR) if f.endswith(".json")]

        for f in files:
            path = os.path.join(QUEUE_DIR, f)

            try:
                with open(path) as jf:
                    job = json.load(jf)

                img_path = job["image"]
                job_id = job["id"]

                img = cv2.imread(img_path)
                if img is None:
                    raise RuntimeError("Image not found")

                inp = preprocess(img)
                output = session.run([output_name], {input_name: inp})

                number, conf = parse_digits(output)

                result = {
                    "job_id": job_id,
                    "raw_number": number,
                    "confidence": conf
                }

                with open(os.path.join(RESULT_DIR, f"{job_id}.json"), "w") as rf:
                    json.dump(result, rf, indent=2)

                print(f"Processed {job_id} → {number}")

            except Exception as e:
                print("ERROR processing job:", f, e)

            finally:
                os.remove(path)

        time.sleep(1)

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    worker_loop()
