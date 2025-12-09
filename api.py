import os
import time
import uuid
import shutil
import logging
import torch
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from color_pipeline import pipeline
from optimization import optimizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="VideoColorGrading API", version="1.0")

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static for downloading results
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

class ProcessResponse(BaseModel):
    processed_video_url: str
    processing_time: float
    used_gpu: str
    quality_mode_used: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}

@app.post("/process", response_model=ProcessResponse)
async def process_video(
    video_file: UploadFile = File(...),
    reference_image: Optional[UploadFile] = File(None),
    quality_mode: str = Form("balanced"), # fast, balanced, high
    stabilization: bool = Form(True),
    output_resolution: str = Form("auto")
):
    request_id = str(uuid.uuid4())
    logger.info(f"Received request {request_id}")
    
    start_time = time.time()
    
    # Save Video
    video_ext = video_file.filename.split('.')[-1]
    video_path = os.path.join(UPLOAD_DIR, f"{request_id}_input.{video_ext}")
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video_file.file, buffer)
        
    # Save Reference if exists
    ref_path = None
    if reference_image:
        ref_ext = reference_image.filename.split('.')[-1]
        ref_path = os.path.join(UPLOAD_DIR, f"{request_id}_ref.{ref_ext}")
        with open(ref_path, "wb") as buffer:
            shutil.copyfileobj(reference_image.file, buffer)
            
    # Output Path
    output_filename = f"{request_id}_output.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        # Run Pipeline
        pipeline.process_video(
            video_path=video_path,
            ref_image_path=ref_path,
            quality_mode=quality_mode,
            stabilization=stabilization,
            output_resolution=output_resolution,
            save_path=output_path
        )
        
        processing_time = time.time() - start_time
        
        # Construct URL (assuming local deployment accessible via same host)
        # In production, upload to S3 and return S3 URL
        processed_video_url = f"/outputs/{output_filename}"
        
        return {
            "processed_video_url": processed_video_url,
            "processing_time": processing_time,
            "used_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "quality_mode_used": quality_mode
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Cleanup Inputs
        if os.path.exists(video_path):
            os.remove(video_path)
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
