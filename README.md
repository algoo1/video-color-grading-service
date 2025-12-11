# Video Color Grading Service (ICCV 2025)

[![RunPod Badge](https://img.shields.io/badge/RunPod-Deploy-blue?logo=runpod)](https://runpod.io/console/deploy?template=algoo1/video-color-grading-service)
[![Docker Image](https://img.shields.io/docker/pulls/algoo1/video-grading.svg)](https://hub.docker.com/r/algoo1/video-grading)

A production-ready SaaS backend engine for **Video Color Grading**, **Correction**, and **Look Matching** using the ICCV 2025 VideoColorGrading model.

It implements a high-performance color grading pipeline using:
- **VideoColorGrading** (ICCV 2025) for LUT generation and style transfer.
- **FastAPI** for the REST interface.
- **PyTorch** with **Mixed Precision (FP16)** and **Torch.compile** optimizations.
- **Decord** for fast video decoding.

## Features

- **Automatic Color Grading**: Learned grading from reference images.
- **Temporal Consistency**: Smooths grading across frames to prevent flickering.
- **Optimized Pipeline**:
  - FP16 Inference
  - Batch Processing
  - GPU-accelerated 3D LUT application (Trilinear Interpolation)
- **Deployment Ready**: Dockerfile and RunPod Serverless handler included.

## Prerequisites

- NVIDIA GPU (A10, A100, H100, or consumer RTX 30/40 series recommended).
- CUDA 12.1+
- Docker (for containerized deployment)

## Setup

### 1. Clone & Install
```bash
git clone <this-repo>
cd video-color-grading-backend
pip install -r requirements.txt
```

### 2. Download Weights
The system requires the pretrained weights for `GS-Extractor` and `L-Diffuser`.
Please download them from the official [VideoColorGrading repository](https://github.com/seunghyuns98/VideoColorGrading) or their provided Google Drive link.

Place the weights in the `pretrained/` directory:
- `pretrained/gs_extractor.pth`
- `pretrained/l_diffuser.pth`

### 3. Run API Server
```bash
python main.py
```
The API will be available at `http://localhost:8000`.

## API Usage

### Process Video
**Endpoint:** `POST /process`

**Parameters:**
- `video_file`: (File) The video to grade.
- `reference_image`: (File, Optional) Image to match the look of. If omitted, uses auto-grading.
- `quality_mode`: (String) `fast`, `balanced`, `high`. Default: `balanced`.
- `stabilization`: (Boolean) Enable temporal smoothing. Default: `true`.

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/process" \
  -F "video_file=@my_video.mp4" \
  -F "reference_image=@cinematic_ref.jpg" \
  -F "quality_mode=balanced"
```

## Deployment

### Docker
```bash
docker build -t video-grading-service .
docker run --gpus all -p 8000:8000 video-grading-service
```

### RunPod Serverless
This project includes a `runpod_handler.py` for serverless deployment.
1. Build the Docker image.
2. Push to a container registry (Docker Hub, GHCR).
3. Create a RunPod Serverless Endpoint using this image.
4. Set the "Docker Command" to `python runpod_handler.py`.

**Input Payload:**
```json
{
  "input": {
    "video_url": "https://example.com/video.mp4",
    "reference_image_url": "https://example.com/ref.jpg",
    "quality_mode": "balanced"
  }
}
```

## Optimizations Implemented

1.  **Trilinear LUT Interpolation**: Custom PyTorch module for applying 3D LUTs on GPU batches, significantly faster than CPU-based application.
2.  **FP16 / AMP**: Automatic Mixed Precision to reduce VRAM usage and increase speed on Tensor Cores.
3.  **Torch Compile**: Enables `torch.compile` (Inductor backend) on supported Linux environments for graph optimization.
4.  **Batch Decoding**: Uses `decord` for efficient frame access.

## Project Structure

- `api.py`: FastAPI application.
- `color_pipeline.py`: Core logic for video processing and grading.
- `model_loader.py`: Manages loading of the VideoColorGrading models.
- `optimization.py`: Helper for GPU device management and compilation.
- `runpod_handler.py`: Entry point for RunPod Serverless.
- `utils.py`: Helper functions for I/O.
