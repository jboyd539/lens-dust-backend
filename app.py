from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import cv2, numpy as np, tempfile, os, ffmpeg

app = FastAPI()

def remove_dust(video_path, threshold=15, inpaint_radius=3, sample_frames=50):
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "cleaned_video.mp4")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    step = max(1, total_frames // sample_frames)
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    median_frame = np.median(frames, axis=0).astype(np.uint8)
    diff = cv2.absdiff(frames[0], median_frame)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret: break
        cleaned = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)
        cv2.imwrite(f"{frame_dir}/frame_{frame_idx:05d}.png", cleaned)
        frame_idx += 1
    cap.release()

    ffmpeg.input(f"{frame_dir}/frame_%05d.png", framerate=fps)\
        .output(output_path, vcodec='libx264', crf=18, pix_fmt='yuv420p')\
        .overwrite_output().run(quiet=True)

    return output_path

@app.post("/clean_video")
async def clean_video(file: UploadFile = File(...), threshold: int = 15, inpaint_radius: int = 3):
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(await file.read())
    temp_input.close()

    cleaned_video = remove_dust(temp_input.name, threshold, inpaint_radius)
    return FileResponse(cleaned_video, media_type="video/mp4", filename="cleaned_video.mp4")
