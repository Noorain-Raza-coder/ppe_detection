from fastapi import FastAPI, UploadFile, File, HTTPException
from src.fastapi_inference import run_pipeline
import numpy as np
import cv2

## build the app
app = FastAPI()

## create the end point
@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ppe-detection")
async def get_ppe_detection(image: UploadFile = File(...)):

    # Step 1: Read the uploaded image as bytes
    image_bytes = await image.read()

    # Step 2: Convert bytes into a NumPy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)

    # Step 3: Decode bytes into an OpenCV image
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}


    # Step 5: Pass the image to your pipeline
    result = run_pipeline(frame)
    return result


## trigger the endpoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
