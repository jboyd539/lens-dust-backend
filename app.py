import os
import uuid
from typing import AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Lens Dust Backend")

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


def generate_unique_filename(ext: str) -> str:
    return f"{uuid.uuid4().hex}.{ext}"


@app.get("/")
async def root():
    return {"message": "Lens Dust Backend Running!"}


@app.post("/preview_dust_mask")
async def preview_dust_mask(file: UploadFile = File(...), threshold: int = Form(15)):
    # Save uploaded file temporarily
    temp_path = os.path.join(UPLOAD_DIR, generate_unique_filename(file.filename.split(".")[-1]))
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Read first frame
    cap = cv2.VideoCapture(temp_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return JSONResponse({"error": "Could not read video frame"}, status_code=400)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 255 - threshold, 255, cv2.THRESH_BINARY)
    mask_path = os.path.join(OUTPUT_DIR, generate_unique_filename("png"))
    cv2.imwrite(mask_path, mask)

    return {"mask_url": f"/outputs/{os.path.basename(mask_path)}"}


@app.post("/clean_video_progress")
async def clean_video_progress(
    file: UploadFile = File(...),
    threshold: int = Form(15),
    inpaint_radius: int = Form(3),
):
    async def process_video() -> AsyncGenerator[str, None]:
        # Save input video
        temp_input_path = os.path.join(UPLOAD_DIR, generate_unique_filename(file.filename.split(".")[-1]))
        temp_output_path = os.path.join(OUTPUT_DIR, generate_unique_filename("mp4"))

        with open(temp_input_path, "wb") as f:
            f.write(await file.read())

        cap = cv2.VideoCapture(temp_input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 255 - threshold, 255, cv2.THRESH_BINARY)
            frame = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)

            out.write(frame)
            frame_idx += 1

            # Send progress
            yield f"data: {{\"frame\": {frame_idx}, \"total\": {total_frames}}}\n\n"

        cap.release()
        out.release()

        yield f"data: {{\"done\": true, \"video_url\": \"/outputs/{os.path.basename(temp_output_path)}\"}}\n\n"

    return StreamingResponse(process_video(), media_type="text/event-stream")
