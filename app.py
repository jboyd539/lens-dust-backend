from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
import uvicorn
import shutil
import os
import cv2
import numpy as np
import uuid
from pathlib import Path
import asyncio
from typing import Optional

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

@app.get("/")
def root():
    return {"message": "Lens Dust Backend Running!"}

@app.post("/clean_video_progress")
async def clean_video_progress(
    file: UploadFile = File(...),
    threshold: int = Form(...),
    inpaint_radius: int = Form(...)
):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    output_path = OUTPUT_DIR / f"{file_id}_cleaned.mp4"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Simulated dust removal: for now, just copy frames
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # TODO: Implement actual dust detection & removal
        # For now, we just pass the frame as-is
        out.write(frame)

        # SSE-like progress output (simulate)
        yield f"data: {JSONResponse({'frame': i+1, 'total': total_frames}).body.decode()}\n\n"
        await asyncio.sleep(0)  # allows async iteration

    cap.release()
    out.release()

    # Send final video URL to frontend
    video_url = f"/download/{output_path.name}"
    yield f"data: {JSONResponse({'done': True, 'video_url': video_url}).body.decode()}\n\n"

@app.get("/download/{filename}")
def download_video(filename: str):
    path = OUTPUT_DIR / filename
    if path.exists():
        return FileResponse(path, media_type="video/mp4", filename=filename)
    return JSONResponse({"error": "File not found"}, status_code=404)

if __name__ == "__main__":
    # Do not use this on Render; Render will use Uvicorn start command
    uvicorn.run(app, host="0.0.0.0", port=10000)
