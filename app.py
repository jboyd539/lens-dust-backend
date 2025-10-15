import os
import uuid
from typing import AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Lens Dust Backend")

# CORS for preview + SSE + static files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # optionally restrict to your app origin
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated assets
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


def generate_unique_filename(ext: str) -> str:
    return f"{uuid.uuid4().hex}.{ext}"


@app.get("/")
async def root():
    return {"message": "Lens Dust Backend Running!"}


@app.post("/preview_dust_mask")
async def preview_dust_mask(
    file: UploadFile = File(...),
    threshold: int = Form(15),
):
    # Basic input validation
    threshold = int(max(0, min(255, threshold)))

    # Save uploaded file temporarily
    ext = (file.filename.split(".")[-1] or "mp4").lower()
    temp_path = os.path.join(UPLOAD_DIR, generate_unique_filename(ext))
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Read first frame
    cap = cv2.VideoCapture(temp_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return JSONResponse({"error": "Could not read video frame"}, status_code=400)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Simple high-intensity thresholding to highlight dust (tweak as needed)
    _, mask = cv2.threshold(gray, 255 - threshold, 255, cv2.THRESH_BINARY)

    mask_path = os.path.join(OUTPUT_DIR, generate_unique_filename("png"))
    cv2.imwrite(mask_path, mask)

    # EXACT field the frontend expects
    return {"mask_url": f"/outputs/{os.path.basename(mask_path)}"}


@app.post("/clean_video_progress")
async def clean_video_progress(
    file: UploadFile = File(...),
    threshold: int = Form(15),
    inpaint_radius: int = Form(3),
):
    # Basic input validation
    threshold = int(max(0, min(255, threshold)))
    inpaint_radius = int(max(1, min(20, inpaint_radius)))

    async def process_video() -> AsyncGenerator[str, None]:
        try:
            # Save input video
            ext = (file.filename.split(".")[-1] or "mp4").lower()
            temp_input_path = os.path.join(UPLOAD_DIR, generate_unique_filename(ext))
            temp_output_path = os.path.join(OUTPUT_DIR, generate_unique_filename("mp4"))

            # Important: read once before iterating
            with open(temp_input_path, "wb") as f:
                f.write(await file.read())

            cap = cv2.VideoCapture(temp_input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

            if total_frames == 0 or width == 0 or height == 0:
                cap.release()
                yield 'data: {"error":"Invalid video input"}\n\n'
                return

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_output_path, fourcc, float(fps), (width, height))

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

                # EXACT format the frontend parses: JSON after "data: "
                yield f'data: {{"frame": {frame_idx}, "total": {total_frames}}}\n\n'

            cap.release()
            out.release()

            # Final event the frontend expects
            final_url = f"/outputs/{os.path.basename(temp_output_path)}"
            yield f'data: {{"done": true, "video_url": "{final_url}"}}\n\n'
        except Exception as e:
            # Send an error event to avoid hanging
            msg = str(e).replace('"', "'")
            yield f'data: {{"error":"{msg}"}}\n\n'

    # SSE headers improve compatibility with some hosts/proxies
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(process_video(), media_type="text/event-stream", headers=headers)
