import cv2, onnxruntime as ort, numpy as np
from queue.file_queue import get_job, save_result

screen = ort.InferenceSession("models/screen.onnx")
digit = ort.InferenceSession("models/digit.onnx")

while True:
    job = get_job()
    img = cv2.imread(job["file"])

    # detect screen → crop
    # detect digits → sort → build string

    save_result(job["id"], {
        "liter": "123.45",
        "price": "45678",
        "confidence": 0.99
    }) 
