from fastapi import FastAPI, UploadFile, File, HTTPException
from src.fastapi_inference import run_pipeline
import numpy as np
import cv2

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ppe-detection")
async def get_ppe_detection(image: UploadFile = File(...)):
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported",
        )

    image_bytes = await image.read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )

    result = run_pipeline(frame)

    if result is None:
        raise HTTPException(
            status_code=500,
            detail="Inference pipeline failed to initialize",
        )

    return result
