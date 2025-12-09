import runpod
import os
import requests
import uuid
import torch
import time
from color_pipeline import pipeline
from optimization import optimizer

# Initialize pipeline once (Cold Start)
print("Initializing Pipeline...")
pipeline.load_resources()
print("Pipeline Initialized.")

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def handler(event):
    """
    RunPod Handler
    Input event: {
        "input": {
            "video_url": "http://...",
            "reference_image_url": "http://... (optional)",
            "quality_mode": "balanced",
            "stabilization": true,
            "output_resolution": "auto"
        }
    }
    """
    job_input = event.get("input", {})
    
    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "No video_url provided"}
        
    ref_url = job_input.get("reference_image_url")
    quality_mode = job_input.get("quality_mode", "balanced")
    stabilization = job_input.get("stabilization", True)
    output_resolution = job_input.get("output_resolution", "auto")
    
    job_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    video_path = os.path.join(temp_dir, "input_video.mp4")
    ref_path = os.path.join(temp_dir, "ref_image.jpg") if ref_url else None
    output_path = os.path.join(temp_dir, "output.mp4")
    
    try:
        # Download Inputs
        print(f"Downloading video from {video_url}")
        download_file(video_url, video_path)
        
        if ref_url:
            print(f"Downloading reference from {ref_url}")
            download_file(ref_url, ref_path)
            
        # Process
        start_time = time.time()
        pipeline.process_video(
            video_path=video_path,
            ref_image_path=ref_path,
            quality_mode=quality_mode,
            stabilization=stabilization,
            output_resolution=output_resolution,
            save_path=output_path
        )
        process_time = time.time() - start_time
        
        # Upload Output (Assuming RunPod Bucket or you return the file bytes/base64 - usually bucket is better)
        # For this template, we will assume the user has a way to upload or we return a presigned URL.
        # But RunPod Serverless often handles the return value. 
        # If we return a file path in some integrations, it uploads it.
        # Here we will just return the path if running locally, or typically upload to S3.
        # Since I don't have S3 credentials, I'll mock the upload or return status.
        # Ideally, we should upload to an S3 bucket provided in env vars.
        
        # For now, let's return a success message and metadata.
        # In a real deployment, you would integrate boto3 here.
        
        return {
            "status": "success",
            "processing_time": process_time,
            "output_path": output_path, # In RunPod, this local path is lost. 
            # TODO: Implement S3 upload
            "message": "Video processed. Configure S3 to upload result." 
        }
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup
        # shutil.rmtree(temp_dir) # Keep for debugging or cleanup
        pass

runpod.serverless.start({"handler": handler})
