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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("remove-bg-api")

app = FastAPI()

# Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request processed in {process_time:.4f} seconds")
    return response

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BiRefNet model and convert to half precision if CUDA is available
birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
birefnet.to(device)
if device == "cuda":
    birefnet.half()
birefnet.eval()

# Smaller transform for faster processing (256×256 instead of 512×512)
transform_image = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_image(image: Image.Image):
    original_size = image.size

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    input_tensor = transform_image(image).unsqueeze(0)
    if device == "cuda":
        input_tensor = input_tensor.to(device, dtype=torch.float16)
    else:
        input_tensor = input_tensor.to(device)

    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid()

    pred_mask = preds[0].squeeze().cpu()
    mask = transforms.ToPILImage()(pred_mask)

    # Resize mask back to the original image size
    mask_resized = mask.resize(original_size)

    # Create RGBA output
    white_mask = Image.new("RGBA", original_size, (255, 255, 255, 0))
    mask_resized = mask_resized.convert("L")
    white_mask.putalpha(mask_resized)

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
        # Use lower-quality WebP for speed trade-off
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

# @app.post("/bulk-processing/")
# async def bulk_processing(file: UploadFile = File(...)):
#     logger.info("Received bulk processing request")
#     start_time = time.time()
#
#     if not file.filename.endswith(".zip"):
#         raise HTTPException(status_code=400, detail="Only ZIP files are accepted")
#
#     zip_file_bytes = BytesIO(await file.read())
#     output_zip = BytesIO()
#
#     # Lower the compression level to speed up
#     with zipfile.ZipFile(zip_file_bytes, "r") as zip_ref:
#         with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_out:
#             for image_name in zip_ref.namelist():
#                 logger.info(f"Processing file: {image_name}")
#
#                 try:
#                     image_data = zip_ref.read(image_name)
#                     input_image = Image.open(BytesIO(image_data)).convert("RGBA")
#                 except:
#                     logger.info(f"Skipping non-image file: {image_name}")
#                     continue
#
#                 try:
#                     mask = await extract_object(input_image)
#
#                     output_bytes = BytesIO()
#                     mask.save(output_bytes, format="WEBP", lossless=True)
#                     output_bytes.seek(0)
#                     zip_out.writestr(f"mask-{image_name}", output_bytes.read())
#                 except Exception as e:
#                     logger.error(f"Error processing {image_name}: {e}")
#
#     output_zip.seek(0)
#     total_time = time.time() - start_time
#     logger.info(f"Bulk processing completed in {total_time:.2f} seconds")
#
#     return StreamingResponse(
#         output_zip,
#         media_type="application/zip",
#         headers={"Content-Disposition": "attachment; filename=processed_images.zip"}
#     )

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
