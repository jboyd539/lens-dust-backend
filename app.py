from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil
import os
import asyncio

app = FastAPI(title="Lens Dust Removal API")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (frontend compatibility)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- File storage setup ---
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
async def root():
    return {"message": "✅ Lens Dust Removal API is running!"}


@app.post("/preview_dust_mask")
async def preview_dust_mask(file: UploadFile = File(...)):
    """
    Receives a video file and returns a dummy preview path.
    Replace with real dust detection logic later.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    preview_path = os.path.join(OUTPUT_DIR, f"preview_{file.filename}")
    shutil.copy(file_path, preview_path)

    return {"preview_path": preview_path}


async def clean_video_generator(file_path: str):
    """
    Simulates a video cleaning process and streams live progress events.
    """
    total_steps = 5
    for step in range(1, total_steps + 1):
        await asyncio.sleep(1)  # simulate work
        progress = round(step / total_steps, 2)
        yield f"data: {{\"progress\": {progress}}}\n\n"

    # when done, return a cleaned video path
    cleaned_path = os.path.join(OUTPUT_DIR, f"cleaned_{os.path.basename(file_path)}")
    shutil.copy(file_path, cleaned_path)
    yield f"data: {{\"done\": \"{cleaned_path}\"}}\n\n"


@app.post("/clean_video_progress")
async def clean_video_progress(file: UploadFile = File(...)):
    """
    Streams real-time progress updates (Server-Sent Events).
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
    }

    return StreamingResponse(
        clean_video_generator(file_path),
        media_type="text/event-stream",
        headers=headers,
    )
