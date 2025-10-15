import os
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from typing import AsyncGenerator

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Lens Dust Backend")

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Lens Dust Backend Running!"}


@app.post("/preview_dust_mask")
async def preview_dust_mask(file: UploadFile = File(...)):
    """
    Generates a preview dust mask for the uploaded video.
    Currently uses a simple threshold to detect bright dust spots.
    """
    video_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}_mask.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Simple thresholding for bright spots (dust)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        out.write(mask)

    cap.release()
    out.release()

    return {"mask_url": f"/outputs/{video_id}_mask.mp4"}


@app.post("/clean_video_progress")
async def clean_video_progress(
    file: UploadFile = File(...),
    threshold: int = Form(15),
    inpaint_radius: int = Form(3),
) -> AsyncGenerator[str, None]:
    """
    Cleans the video of dust spots frame by frame.
    Sends progress updates via Server-Sent Events (SSE)
    """

    video_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}_cleaned.mp4")

    # Save uploaded video
    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    async def event_stream() -> AsyncGenerator[str, None]:
        nonlocal frame_idx
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect bright dust pixels
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            # Inpaint to remove dust
            cleaned_frame = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)
            out.write(cleaned_frame)

            frame_idx += 1
            yield f"data: { { 'frame': frame_idx, 'total': total_frames } }\n\n"

        cap.release()
        out.release()
        yield f"data: { { 'done': True, 'video_url': f'/outputs/{video_id}_cleaned.mp4' } }\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    return StreamingResponse(event_stream(), headers=headers)


@app.get("/outputs/{filename}")
async def get_output(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"error": "File not found"})
