FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the VideoColorGrading repository
RUN git clone https://github.com/seunghyuns98/VideoColorGrading.git

# Install repository specific requirements
RUN pip install -r VideoColorGrading/requirements.txt || true

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install repository specific requirements if any (often in the repo)
# RUN pip install -r VideoColorGrading/requirements.txt || true

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads outputs pretrained

# Download pretrained models (You should provide a script or instruction for this)
# For now, we assume they are mounted or downloaded manually.

# Expose API port
EXPOSE 8000

# Default command: Run the API
CMD ["python", "main.py"]
