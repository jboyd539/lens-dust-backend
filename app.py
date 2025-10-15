import os
import cv2
import numpy as np
import tempfile
import shutil
import ffmpeg
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Generator
import asyncio
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "processed_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "Lens Dust Backend Running!"}


# ----------------------
# 🟠 PREVIEW ENDPOINT
# ----------------------
@app.post("/preview_dust_mask")
async def preview_dust_mask(file: UploadFile = File(...), threshold: int = Form(15)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JSONResponse({"error": "Failed to read video frame"}, status_code=400)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    diff = cv2.absdiff(gray, blur)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Highlight detected regions in red overlay
    overlay = frame.copy()
    overlay[mask > 0] = [0, 0, 255]
    preview = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    preview_path = os.path.join(OUTPUT_DIR, f"preview_{os.path.basename(tmp_path)}.jpg")
    cv2.imwrite(preview_path, preview)

    return FileResponse(preview_path, media_type="image/jpeg")


# ----------------------
# 🟢 CLEAN VIDEO ENDPOINT
# ----------------------
@app.post("/clean_video_progress")
async def clean_video_progress(file: UploadFile = File(...), threshold: int = Form(15), inpaint_radius: int = Form(3)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cleaned_path = os.path.join(OUTPUT_DIR, f"cleaned_{os.path.basename(tmp_path)}")
    out = cv2.VideoWriter(cleaned_path, fourcc, fps, (width, height))

    # Build average dust mask
    avg_frame = np.zeros((height, width), dtype=np.float32)
    for i in range(0, total_frames, max(1, total_frames // 20)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_frame += gray
    avg_frame /= max(1, total_frames // 20)

    avg_blur = cv2.medianBlur(avg_frame.astype(np.uint8), 7)
    diff = cv2.absdiff(avg_frame.astype(np.uint8), avg_blur)
    _, dust_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    async def process_frames() -> Generator[bytes, None, None]:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Optical flow motion compensation (simple)
            if frame_count > 0:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                h, w = flow.shape[:2]
                flow_map = np.column_stack((np.repeat(np.arange(w), h), np.tile(np.arange(h), w)))
                flow_map = flow_map.reshape(h, w, 2).astype(np.float32)
                flow_map += flow
                frame = cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)

            prev_frame = frame.copy()
            inpainted = cv2.inpaint(frame, dust_mask.astype(np.uint8), inpaint_radius, cv2.INPAINT_TELEA)
            out.write(inpainted)
            frame_count += 1

            yield f"data: {json.dumps({'frame': frame_count, 'total': total_frames})}\n\n".encode()
            await asyncio.sleep(0.001)

        cap.release()
        out.release()

        # Generate thumbnail
        thumb_path = cleaned_path.replace(".mp4", "_thumb.jpg")
        os.system(f"ffmpeg -y -i {cleaned_path} -vf 'thumbnail,scale=320:-1' -frames:v 1 {thumb_path}")

        final_data = json.dumps({
            "done": True,
            "video_url": f"/download/{os.path.basename(cleaned_path)}",
            "thumbnail_url": f"/download/{os.path.basename(thumb_path)}"
        })
        yield f"data: {final_data}\n\n".encode()

    return StreamingResponse(process_frames(), media_type="text/event-stream")


# ----------------------
# 🔵 DOWNLOAD ENDPOINT
# ----------------------
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    if filename.endswith(".jpg"):
        return FileResponse(file_path, media_type="image/jpeg")
    return FileResponse(file_path, media_type="video/mp4")
