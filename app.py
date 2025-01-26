import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from io import BytesIO
import logging
import zipfile
import time
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import asyncio

# 1) Import the NVML bindings from nvidia-ml-py3 (pip install nvidia-ml-py3)
import nvidia_smi

##############################################################################
# ENV VAR to control logging
##############################################################################
verbose_env = os.environ.get("VERBOSE_LOGGING", "true").lower()
if verbose_env == "true":
    LOG_LEVEL = logging.INFO
else:
    LOG_LEVEL = logging.ERROR

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("remove-bg-api")

##############################################################################
# FastAPI setup
##############################################################################
app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # If not in verbose mode, we won't log requests at the INFO level
    logger.info(f"Incoming request: {request.method} {request.url}")
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request processed in {process_time:.4f} seconds")
    return response


device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    nvidia_smi.nvmlInit()
    logger.info("NVML initialized for GPU utilization monitoring.")

##############################################################################
# Load model
##############################################################################
birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
birefnet.to(device)
if device == "cuda":
    birefnet.half()
birefnet.eval()

##############################################################################
# Cache GPU usage every 5 seconds
##############################################################################
_last_gpu_check = 0.0
_cached_usage = 0
CHECK_INTERVAL = 5.0


def get_gpu_usage_percent_cached() -> int:
    """Return GPU usage (0-100), only checking NVML every 5s."""
    global _last_gpu_check, _cached_usage

    if device != "cuda":
        return 0

    now = time.time()
    if (now - _last_gpu_check) > CHECK_INTERVAL:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        _cached_usage = util.gpu
        _last_gpu_check = now
        logger.info(f"Refreshed GPU usage => {_cached_usage}%")
    return _cached_usage


def choose_resize_dim() -> int:
    """Pick a dynamic dimension from [256,384,512,768,1024] based on GPU usage."""
    if device != "cuda":
        return 512

    usage_percent = get_gpu_usage_percent_cached()
    if usage_percent < 20:
        return 1024
    elif usage_percent < 40:
        return 768
    elif usage_percent < 60:
        return 512
    elif usage_percent < 80:
        return 384
    else:
        return 256


def process_image(image: Image.Image):
    overall_start = time.time()

    step_start = time.time()
    original_size = image.size
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    logger.info(f"Step 1 (Image mode check/convert) took {time.time() - step_start:.4f}s")

    step_start = time.time()
    dynamic_dim = choose_resize_dim()
    logger.info(f"GPU usage => adjusting transform to {dynamic_dim}x{dynamic_dim}")
    logger.info(f"Step 2 (choose_resize_dim) took {time.time() - step_start:.4f}s")

    step_start = time.time()
    dynamic_transform = transforms.Compose([
        transforms.Resize((dynamic_dim, dynamic_dim)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = dynamic_transform(image).unsqueeze(0)
    logger.info(f"Step 3 (image transform) took {time.time() - step_start:.4f}s")

    step_start = time.time()
    if device == "cuda":
        input_tensor = input_tensor.to(device, dtype=torch.float16)
    else:
        input_tensor = input_tensor.to(device)
    logger.info(f"Step 4 (move to device) took {time.time() - step_start:.4f}s")

    step_start = time.time()
    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid()
    logger.info(f"Step 5 (model inference) took {time.time() - step_start:.4f}s")

    step_start = time.time()
    pred_mask = preds[0].squeeze().cpu()
    mask = transforms.ToPILImage()(pred_mask)
    logger.info(f"Step 6 (convert mask to PIL) took {time.time() - step_start:.4f}s")

    step_start = time.time()
    mask_resized = mask.resize(original_size)
    white_mask = Image.new("RGBA", original_size, (255, 255, 255, 0))
    mask_resized = mask_resized.convert("L")
    white_mask.putalpha(mask_resized)
    logger.info(f"Step 7 (resize mask/apply alpha) took {time.time() - step_start:.4f}s")

    logger.info(f"Total process_image() time: {time.time() - overall_start:.4f}s")
    return white_mask


async def extract_object(image: Image.Image):
    return await asyncio.to_thread(process_image, image)


@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(None)):
    logger.info("Received request to remove background")
    if file is None:
        raise HTTPException(status_code=400, detail="No image file provided")

    try:
        input_image = Image.open(file.file).convert("RGBA")
        mask = await extract_object(input_image)

        output_bytes = BytesIO()
        mask.save(output_bytes, format="WEBP", lossless=True)
        output_bytes.seek(0)
        return StreamingResponse(
            output_bytes,
            media_type="image/webp",
            headers={"Content-Disposition": "attachment; filename=mask.webp"}
        )
    except Exception as e:
        logger.error(f"Error processing the image: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image.")


@app.options("/{path:path}")
async def preflight_handler(request: Request, path: str):
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
        }
    )


@app.on_event("shutdown")
def shutdown_event():
    if device == "cuda":
        logger.info("Shutting down NVML.")
        nvidia_smi.nvmlShutdown()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
