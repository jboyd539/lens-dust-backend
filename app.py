import os
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator

app = FastAPI(title="Lens Dust Backend")

# CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_unique_filename(extension="mp4"):
    return f"{uuid.uuid4().hex}.{extension}"

@app.get("/")
async def root():
    return {"message": "Lens Dust Backend Running!"}

@app.post("/preview_dust_mask")
async def preview_dust_mask(file: UploadFile = File(...)):
    """
    Generates a preview mask highlighting areas of dust/smears on the first frame.
    """
    temp_path = os.path.join(UPLOAD_DIR, generate_unique_filename(file.filename.split(".")[-1]))
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(temp_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return JSONResponse({"error": "Failed to read video"}, status_code=400)

    # Dummy mask generation for preview purposes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # highlight bright spots
    mask_path = os.path.join(OUTPUT_DIR, generate_unique_filename("png"))
    cv2.imwrite(mask_path, mask)

    return {"mask_url": f"/outputs/{os.path.basename(mask_path)}"}

@app.post("/clean_video_progress", response_model=None)
async def clean_video_progress(
    file: UploadFile = File(...),
    threshold: int = Form(15),
    inpaint_radius: int = Form(3),
) -> AsyncGenerator[str, None]:
    """
    Cleans lens dust marks from video frames, streams progress via SSE,
    and returns a video URL when finished.
    """
    temp_input_path = os.path.join(UPLOAD_DIR, generate_unique_filename(file.filename.split(".")[-1]))
    temp_output_path = os.path.join(OUTPUT_DIR, generate_unique_filename("mp4"))

    # Save uploaded file
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

        # Dummy dust removal: in real implementation, replace with AI or inpainting
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        frame = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)

        out.write(frame)
        frame_idx += 1

        # Yield SSE progress
        yield f"data: {{" \
              f"\"frame\": {frame_idx}, " \
              f"\"total\": {total_frames}" \
              f"}}\n\n"

    cap.release()
    out.release()

    # Send final event with video URL
    yield f"data: {{\"done\": true, \"video_url\": \"/outputs/{os.path.basename(temp_output_path)}\"}}\n\n"

# Serve static files from outputs directory
from fastapi.staticfiles import StaticFiles
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
