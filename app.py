# app.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import numpy as np
import cv2
import tempfile
import shutil
import os
import json
import asyncio

app = FastAPI()

# === Enable CORS for Lovable frontend ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Dust removal helper function ===
def remove_dust(frame, threshold=15, inpaint_radius=3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cleaned = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned

# === Endpoint: direct download (no progress) ===
@app.post("/clean_video")
async def clean_video(
    file: UploadFile = File(...),
    threshold: int = Form(15),
    inpaint_radius: int = Form(3)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        shutil.copyfileobj(file.file, temp_input)
        input_path = temp_input.name

    output_path = input_path.replace(".mp4", "_cleaned.mp4")

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cleaned_frame = remove_dust(frame, threshold, inpaint_radius)
        out.write(cleaned_frame)

    cap.release()
    out.release()

    return FileResponse(output_path, filename="cleaned.mp4", media_type="video/mp4")

# === Endpoint: SSE progress + final video ===
@app.post("/clean_video_progress")
async def clean_video_progress(
    file: UploadFile = File(...),
    threshold: int = Form(15),
    inpaint_radius: int = Form(3)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        shutil.copyfileobj(file.file, temp_input)
        input_path = temp_input.name

    # Save final video in /tmp so it can be served
    output_path = f"/tmp/{os.path.basename(input_path).replace('.mp4', '_cleaned.mp4')}"

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    async def event_stream():
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cleaned_frame = remove_dust(frame, threshold, inpaint_radius)
            out.write(cleaned_frame)
            frame_num += 1
            yield f"data: {json.dumps({'frame': frame_num, 'total': total_frames})}\n\n"
            await asyncio.sleep(0)

        cap.release()
        out.release()
        yield f"data: {json.dumps({'done': True, 'video_url': f'/download/{os.path.basename(output_path)}'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# === Endpoint: serve final video ===
@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = f"/tmp/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename="cleaned.mp4", media_type="video/mp4")
    return {"error": "File not found"}

# === Run with dynamic Render port ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),  # Use Render's port
        reload=True
    )
