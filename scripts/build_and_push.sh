#!/bin/bash

# Configuration
IMAGE_NAME="video-grading"
DOCKER_USERNAME="" # Enter your DockerHub username here or pass as argument
TAG="v1"

# Check if Docker Username is provided
if [ -z "$1" ] && [ -z "$DOCKER_USERNAME" ]; then
    echo "Error: Docker Hub username is required."
    echo "Usage: ./build_and_push.sh <your_dockerhub_username>"
    exit 1
fi

if [ -n "$1" ]; then
    DOCKER_USERNAME=$1
fi

FULL_IMAGE_NAME="$DOCKER_USERNAME/$IMAGE_NAME:$TAG"

echo "=================================================="
echo "  VideoColorGrading Build & Push Script"
echo "  Target Image: $FULL_IMAGE_NAME"
echo "=================================================="

# 1. Install Docker if missing (RunPod templates usually have it, but just in case)
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get update && apt-get install -y docker.io
fi

# 2. Download Weights (If they don't exist)
# NOTE: You must provide valid URLs or upload them manually to 'pretrained/' before running this
if [ ! -f "pretrained/gs_extractor.pth" ] || [ ! -f "pretrained/l_diffuser.pth" ]; then
    echo "WARNING: Model weights not found in pretrained/ directory."
    echo "For the Docker image to work, you typically need to include these weights."
    echo "Please upload 'gs_extractor.pth' and 'l_diffuser.pth' to the 'pretrained/' folder."
    read -p "Do you want to continue anyway (weights will be missing)? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 3. Login to Docker Hub
echo "Logging in to Docker Hub..."
docker login

# 4. Build Image
echo "Building Docker Image..."
docker build -t $FULL_IMAGE_NAME .

# 5. Push Image
echo "Pushing to Docker Hub..."
docker push $FULL_IMAGE_NAME

echo "=================================================="
echo "  SUCCESS!"
echo "  Image pushed to: $FULL_IMAGE_NAME"
echo "=================================================="
echo "  Now go to RunPod Serverless -> New Endpoint"
echo "  Container Image: $FULL_IMAGE_NAME"
echo "  Docker Command: python runpod_handler.py"
echo "=================================================="
