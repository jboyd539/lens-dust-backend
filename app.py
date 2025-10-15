# api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import uuid
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your frontend URL if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
PROCESSED_DIR = "./processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


@app.post("/preview_dust_mask")
async def preview_dust_mask(file: UploadFile = File(...)):
    """
    Accepts a video, analyzes dust/smears, and returns a preview mask URL.
    """
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # TODO: Implement dust detection & generate mask image
    # For now, generate a fake mask URL (replace with actual processing)
    preview_mask_url = f"https://yourcdn.com/masks/{file_id}_mask.png"

    return {"mask_url": preview_mask_url}


@app.post("/clean_video_progress")
async def clean_video_progress(file: UploadFile = File(...)):
    """
    Accepts a video, cleans dust, and streams progress via SSE.
    Returns a final JSON with `done: true` and `video_url`.
    """
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(input_path, "wb") as f:
        f.write(await file.read())

    output_path = os.path.join(PROCESSED_DIR, f"{file_id}_cleaned.mp4")

    async def progress_generator():
        """
        Yield SSE numeric progress, then final JSON with video URL
        """
        for progress in range(0, 101, 5):
            await asyncio.sleep(0.1)  # simulate processing delay
            yield f"data: {progress}\n\n"

        # After processing, yield final JSON with video URL
        video_url = f"https://yourcdn.com/processed/{file_id}_cleaned.mp4"
        final_event = {"done": True, "video_url": video_url}
        yield f"data: {final_event}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "text/event-stream",
    }

    return StreamingResponse(progress_generator(), headers=headers)
