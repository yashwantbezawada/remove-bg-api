import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from io import BytesIO
import logging
import time
import torch

from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import torch.nn.functional as F
import asyncio

# 1) Import the NVML bindings from nvidia-ml-py3
import nvidia_smi

##############################################################################
# ENV VAR to control logging
##############################################################################
verbose_env = os.environ.get("VERBOSE_LOGGING", "false").lower()
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
# GPU usage caching logic (unchanged)
##############################################################################
_last_gpu_check = 0.0
_cached_usage = 0
CHECK_INTERVAL = 5.0

def get_gpu_usage_percent_cached() -> int:
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

##############################################################################
# MAIN: GPU-based resizing + single-step alpha creation
##############################################################################
def process_image(image: Image.Image):
    overall_start = time.time()

    step_start = time.time()
    original_width, original_height = image.size

    # Convert RGBA -> RGB if needed
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    logger.info(f"Step 1 (Image mode check/convert) took {time.time() - step_start:.4f}s")

    step_start = time.time()
    dynamic_dim = choose_resize_dim()
    logger.info(f"GPU usage => adjusting transform to {dynamic_dim}x{dynamic_dim}")
    logger.info(f"Step 2 (choose_resize_dim) took {time.time() - step_start:.4f}s")

    step_start = time.time()
    # Basic transform => [1,3,H,W]
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
        preds = birefnet(input_tensor)[-1].sigmoid()  # shape [1,1,H',W']
    logger.info(f"Step 5 (model inference) took {time.time() - step_start:.4f}s")

    step_start = time.time()
    # preds => [1,1,H',W']
    # GPU-based mask resize -> original shape
    #   Convert to [N,C,H,W] before calling interpolate
    mask_small_gpu = preds[0]  # shape [1,H',W']
    mask_small_gpu = mask_small_gpu.unsqueeze(0)  # shape [1,1,H',W']
    mask_resized_gpu = F.interpolate(
        mask_small_gpu,
        size=(original_height, original_width),
        mode='nearest'  # or "bilinear" if you prefer
    )  # shape => [1,1,origH,origW]

    # Now shape => [1,1,origH,origW], float16 in [0..1]
    # Convert to 8-bit alpha on GPU
    mask_alpha_8_gpu = (mask_resized_gpu * 255).clamp(0, 255).to(torch.uint8)  # [1,1,origH,origW]

    # Create an RGBA 8-bit tensor: R=255, G=255, B=255, alpha=mask
    # shape => [1,4,origH,origW]
    # We'll do it all on GPU
    white_8_gpu = torch.full(
        (1, 3, original_height, original_width), fill_value=255, dtype=torch.uint8, device=device
    )
    rgba_gpu = torch.cat([white_8_gpu, mask_alpha_8_gpu], dim=1)

    # Now shape => [1,4,origH,origW], 8-bit
    # Move to CPU as a NumPy array
    rgba_np = rgba_gpu.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # shape => [origH, origW, 4]

    logger.info(f"Step 6 (GPU-based resize & RGBA creation) took {time.time() - step_start:.4f}s")

    step_start = time.time()
    # Single Pillow creation
    final_image = Image.fromarray(rgba_np, mode="RGBA")
    logger.info(f"Step 7 (PIL creation) took {time.time() - step_start:.4f}s")

    logger.info(f"Total process_image() time: {time.time() - overall_start:.4f}s")
    return final_image


async def extract_object(image: Image.Image):
    return await asyncio.to_thread(process_image, image)

@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(None)):
    logger.info("Received request to remove background")
    if file is None:
        raise HTTPException(status_code=400, detail="No image file provided")

    try:
        input_image = Image.open(file.file).convert("RGBA")
        final_img = await extract_object(input_image)

        output_bytes = BytesIO()
        # Save final_img to WebP (lossless or not, your choice)
        final_img.save(output_bytes, format="WEBP", lossless=True)
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
