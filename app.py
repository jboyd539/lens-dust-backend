# api.py
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Lens Dust Backend")

# -----------------------
# Enable CORS
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Preview Dust Mask
# -----------------------
@app.post("/preview_dust_mask")
async def preview_dust_mask(file: UploadFile = File(...)):
    """
    Receive a video, analyze it, and return a preview JSON of the dust mask.
    """
    # Placeholder logic: return a dummy mask summary
    return JSONResponse(
        content={
            "filename": file.filename,
            "preview_mask": [
                {"frame": 1, "dust_coords": [(100, 200), (150, 250)]},
                {"frame": 2, "dust_coords": [(105, 205), (155, 255)]},
            ],
        }
    )

# -----------------------
# Clean Video Progress (SSE)
# -----------------------
@app.post("/clean_video_progress")
async def clean_video_progress(file: UploadFile = File(...)):
    """
    Simulate cleaning the video, returning a streaming SSE response
    showing progress from 0 to 100%.
    """

    def progress_generator():
        total_steps = 100
        for i in range(total_steps + 1):
            # Yield SSE data format: "data: <value>\n\n"
            yield f"data: {i}\n\n"
            time.sleep(0.05)  # simulate processing time

    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


# -----------------------
# Healthcheck / Root
# -----------------------
@app.get("/")
async def root():
    return {"message": "Lens Dust Backend Running!"}
