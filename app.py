# app.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tempfile
import shutil
import os

app = FastAPI()

# === Enable CORS so Lovable frontend can call this API ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your Lovable frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Helper function to remove dust from a frame ===
def remove_dust(frame, threshold=15, inpaint_radius=3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cleaned = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cleaned

# === Endpoint to clean video ===
@app.post("/clean_video")
async def clean_video(
    file: UploadFile = File(...),
    threshold: int = Form(15),
    inpaint_radius: int = Form(3)
):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        shutil.copyfileobj(file.file, temp_input)
        input_path = temp_input.name

    # Output temporary file
    output_path = input_path.replace(".mp4", "_cleaned.mp4")

    # Read video
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
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

    # Return cleaned video
    return_file = open(output_path, "rb")
    data = return_file.read()
    return_file.close()

    # Clean up temp files
    os.remove(input_path)
    os.remove(output_path)

    return data
