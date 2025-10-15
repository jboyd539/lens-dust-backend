from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import tempfile
import cv2
import numpy as np
import json
import os
import asyncio

app = FastAPI()

# Allow Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace * with your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Lens Dust Backend Running!"}


@app.post("/clean_video_progress")
async def clean_video_progress(
    file: UploadFile = File(...),
    threshold: int = Form(15),
    inpaint_radius: int = Form(3)
):
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(await file.read())
    temp_input.close()

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output.close()

    cap = cv2.VideoCapture(temp_input.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not fps or fps <= 0:
        fps = 30.0  # default if FPS can't be read

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    async def event_generator():
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
            cleaned = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)
            out.write(cleaned)

            frame_num += 1
            progress_data = json.dumps({"frame": frame_num, "total": total_frames})
            yield f"data: {progress_data}\n\n"

            # Important: forces Render to flush output as it happens
            await asyncio.sleep(0.01)

        cap.release()
        out.release()

        # Send final SSE event with download URL
        final_data = json.dumps({
            "done": True,
            "video_url": f"/download/{os.path.basename(temp_output.name)}"
        })
        yield f"data: {final_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Endpoint to serve finished videos"""
    path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(path):
        return {"error": "File not found"}
    return FileResponse(path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
