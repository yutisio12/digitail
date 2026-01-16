from fastapi import FastAPI, UploadFile
import uuid, shutil
from queue.file_queue import enqueue, get_result

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile):
    job_id = str(uuid.uuid4())
    path = f"uploads/{job_id}.jpg"

    with open(path,"wb") as f:
        shutil.copyfileobj(file.file, f)

    enqueue(job_id, path)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
def status(job_id: str):
    return {"done": get_result(job_id) is not None}

@app.get("/result/{job_id}")
def result(job_id: str):
    return get_result(job_id) 
