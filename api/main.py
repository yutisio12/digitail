from fastapi import FastAPI, UploadFile
import os, shutil
from api.job_queue import enqueue

app = FastAPI()
os.makedirs("uploads", exist_ok=True)

@app.post("/upload")
async def upload(file: UploadFile):
    path = f"uploads/{file.filename}"

    with open(path,"wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = enqueue(path)
    return {"job_id": job_id}