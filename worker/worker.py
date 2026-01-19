import time, os, json, cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/best2.onnx"
QUEUE_DIR = "queue"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=ort.get_available_providers()
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = img[:, :, ::-1] / 255.0
    img = np.transpose(img, (2,0, 1))
    return np.expand_dims(img.astype(np.float32), 0)

def worker_loop():
    while True:
        for f in os.listdir(QUEUE_DIR):
            if not f.endswith(".json"):
                continue

            path = os.path.join(QUEUE_DIR, f)
            with open(path) as jf:
                job = json.load(jf)

            img = cv2.imread(job["image"])
            inp = preprocess(img)
            output = session.run([output_name], {input_name: inp})

            print("Processed:", job["id"])
            os.remove(path)         

        time.sleep(1) 

if __name__ == "__main__":
    worker_loop() 
            