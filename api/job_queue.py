import json, uuid, os

QUEUE_DIR = "queue"
os.makedirs(QUEUE_DIR, exist_ok=True)

def enqueue(image_path):
  job_id = str(uuid.uuid4())
  with open(f"{QUEUE_DIR}/{job_id}.json", "w") as f:
    json.dump({"id": job_id, "image": image_path}, f)
  return job_id