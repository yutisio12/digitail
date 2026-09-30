from fastapi import FastAPI, UploadFile
import os, shutil
from api.job_queue import enqueue

app = FastAPI()
os.makedirs("upload", exist_ok=True)

@app.post("/uploads")
async def upload(file: UploadFile):
    path = f"uploads/{file.filename}"

    with open(path,"wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = enqueue(path)
    return {"job_id": job_id}
