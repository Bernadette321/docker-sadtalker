from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from app.sadtalker.inference import run_sadtalker
import tempfile
import shutil
import os
from uuid import uuid4

app = FastAPI()

@app.post("/generate")
async def generate_video(image: UploadFile = File(...), audio: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, f"{uuid4().hex}.jpg")
    audio_path = os.path.join(temp_dir, f"{uuid4().hex}.wav")

    with open(image_path, "wb") as img_file:
        img_file.write(await image.read())
    with open(audio_path, "wb") as aud_file:
        aud_file.write(await audio.read())

    try:
        final_video = run_sadtalker(image_path, audio_path)
        return FileResponse(final_video, media_type="video/mp4", filename="talking_baby.mp4")
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)