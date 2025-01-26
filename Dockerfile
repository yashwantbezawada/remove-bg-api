# 1) Use an NVIDIA runtime image. Here we pick CUDA 11.8, Ubuntu 22.04, "runtime" variant.
#    This includes minimal CUDA libraries, NVML (for nvidia-smi), etc.
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 2) Install system packages needed for Pillow or pillow-simd (zlib, jpeg, build tools).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-setuptools python3-dev \
    zlib1g-dev libjpeg-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3) Create an /app dir and copy in your requirements.
WORKDIR /app
COPY requirements.txt /app/

# 4) Install requirements (including pillow-simd and anything else).
RUN pip3 install --no-cache-dir -r requirements.txt

# 5) Install your specific PyTorch + CUDA wheels, as you had before.
#    Make sure this matches the CUDA version (11.8 in this example).
RUN pip3 install \
  torch==2.4.0+cu118 \
  torchvision==0.19.0+cu118 \
  torchaudio==2.4.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# 6) Copy your FastAPI code (app.py) into the container
COPY app.py /app/

# 7) Expose port 8000 for the API
EXPOSE 8000

# 8) Run Gunicorn with Uvicorn workers at container start
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:8000"]
