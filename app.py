from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import logging
import asyncio

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BiRefNet model from HuggingFace
birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
birefnet.to(device)
birefnet.eval()

# Define image transformation (Expecting RGB images)
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Reduced size to 512x512 for faster processing
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# The actual image processing logic is kept synchronous but handled in a separate thread
def process_image(image: Image.Image):
    original_size = image.size

    # Convert the image to RGB if it's in RGBA format
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Apply transformations and keep in float32 (default)
    input_tensor = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()

    pred_mask = preds[0].squeeze()
    mask = transforms.ToPILImage()(pred_mask)

    mask_resized = mask.resize(original_size)

    white_mask = Image.new("RGBA", original_size, (255, 255, 255, 0))
    mask_resized = mask_resized.convert("L")
    white_mask.putalpha(mask_resized)

    return white_mask

# Function to extract the object mask and resize to original image size
async def extract_object(image: Image.Image):
    # Run the processing in a separate thread using asyncio.to_thread()
    return await asyncio.to_thread(process_image, image)

@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(None), image_url: str = Form(None)):
    try:
        if file is not None:
            try:
                # Open the uploaded image
                input_image = Image.open(file.file)

                # Convert the image to RGBA for consistency in the output later
                input_image = input_image.convert("RGBA")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        else:
            raise HTTPException(status_code=400, detail="No image file or URL provided")

        # Generate mask for the image
        mask = await extract_object(input_image)

        # Save the result as WebP format
        output_bytes = BytesIO()
        mask.save(output_bytes, format="WEBP", lossless=True)
        output_bytes.seek(0)

        return StreamingResponse(output_bytes, media_type="image/webp",
                                 headers={"Content-Disposition": "attachment; filename=mask.webp"})
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)