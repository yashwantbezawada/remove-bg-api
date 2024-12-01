import cv2
import zipfile
import os
import numpy as np
from io import BytesIO
import requests

# Constants
VIDEO_FILE = "test_video.mp4"
ZIP_FILE = "frames.zip"
PROCESSED_ZIP_FILE = "processed_frames.zip"
PROCESSED_FRAMES_DIR = "processed_frames"
OUTPUT_VIDEO_FILE = "mask_video.mp4"
MASKED_VIDEO_FILE = "masked_output.mp4"
BULK_PROCESSING_URL = "http://localhost:8000/bulk-processing/"
DOWNSCALE_FACTOR = 0.75  # Downscale frames by 25% (set to 1 to avoid downscaling)
JPEG_QUALITY = 50  # JPEG quality (1-100, higher means better quality but larger size)


def video_to_frames(video_file, output_dir):
    """
    Extracts all frames from the video and saves them as individual image files.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise Exception(f"Failed to open video file: {video_file}")

    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit when there are no more frames

        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_file}")
    return output_dir


def downscale_frame(frame, scale=DOWNSCALE_FACTOR):
    if scale == 1:  # No downscaling
        return frame, frame.shape[:2][::-1]
    height, width = frame.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA), (width, height)


def zip_frames_downscaled(frame_dir, zip_file_name):
    original_sizes = {}
    with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(frame_dir):
            for file in files:
                file_path = os.path.join(root, file)
                frame = cv2.imread(file_path)
                downscaled_frame, original_size = downscale_frame(frame)
                original_sizes[file] = original_size

                # Write downscaled frame as JPEG to the zip
                _, buffer = cv2.imencode(".jpg", downscaled_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                zipf.writestr(file, buffer.tobytes())
    return original_sizes


def unzip_and_upscale(zip_file_path, output_dir, original_sizes):
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, "r") as zipf:
        for file_name in zipf.namelist():
            original_file_name = file_name[len("mask-"):] if file_name.startswith("mask-") else file_name

            if original_file_name not in original_sizes:
                raise KeyError(f"Original size for {original_file_name} not found in original_sizes.")

            frame_data = zipf.read(file_name)
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_UNCHANGED)
            upscaled_frame = cv2.resize(frame, original_sizes[original_file_name], interpolation=cv2.INTER_LINEAR)
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, upscaled_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])


def send_zip_to_endpoint(zip_file_path, url):
    with open(zip_file_path, "rb") as zip_buffer:
        files = {"file": ("frames.zip", zip_buffer, "application/zip")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        with open(PROCESSED_ZIP_FILE, "wb") as f:
            f.write(response.content)
        print(f"Processed frames saved to {PROCESSED_ZIP_FILE}")
    else:
        print(f"Failed to process frames: {response.status_code}, {response.text}")


def stitch_frames_to_video(frames_dir, output_video_file):
    """
    Stitches all frames in the directory into a video at 30 FPS.
    """
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith((".png", ".jpg", ".jpeg"))])

    if not frame_files:
        raise ValueError("No frames found to stitch into a video.")

    # Read the first frame to determine video dimensions
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise ValueError("Failed to read the first frame to determine video dimensions.")

    height, width, _ = first_frame.shape
    fps = 30  # Fixed frame rate

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}. Skipping...")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_video_file}")


def apply_mask_to_video(original_video_file, mask_video_file, output_video_file):
    """
    Applies a mask video on top of the original video. Only the white parts of the mask show
    the corresponding parts of the original video, while the rest of the background is black.
    """
    # Open the original video and mask video
    original_cap = cv2.VideoCapture(original_video_file)
    mask_cap = cv2.VideoCapture(mask_video_file)

    if not original_cap.isOpened():
        raise Exception(f"Failed to open original video: {original_video_file}")
    if not mask_cap.isOpened():
        raise Exception(f"Failed to open mask video: {mask_video_file}")

    # Get video properties
    width = int(original_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Fixed FPS

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    while True:
        ret_original, original_frame = original_cap.read()
        ret_mask, mask_frame = mask_cap.read()

        if not ret_original or not ret_mask:
            break  # Exit when either video ends

        # Convert the mask frame to grayscale
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

        # Create a binary mask where white areas are 255 and everything else is 0
        _, binary_mask = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)

        # Apply the mask to the original frame
        masked_frame = cv2.bitwise_and(original_frame, original_frame, mask=binary_mask)

        # Write the masked frame to the output video
        video_writer.write(masked_frame)

    # Release resources
    original_cap.release()
    mask_cap.release()
    video_writer.release()
    print(f"Masked video saved as {output_video_file}")


def main():
    # Step 1: Convert video to frames
    frames_dir = "frames"
    print("Extracting frames from video...")
    video_to_frames(VIDEO_FILE, frames_dir)

    # Step 2: Downscale and zip the frames
    print("Downscaling frames and zipping them...")
    original_sizes = zip_frames_downscaled(frames_dir, ZIP_FILE)

    # Step 3: Send the zip file to the endpoint
    print("Sending zip file to bulk-processing endpoint...")
    send_zip_to_endpoint(ZIP_FILE, BULK_PROCESSING_URL)

    # Step 4: Unzip and upscale the processed frames
    print("Upscaling processed frames back to original sizes...")
    unzip_and_upscale(PROCESSED_ZIP_FILE, PROCESSED_FRAMES_DIR, original_sizes)

    # Step 5: Stitch processed frames into a video
    print("Stitching processed frames into a video...")
    stitch_frames_to_video(PROCESSED_FRAMES_DIR, OUTPUT_VIDEO_FILE)

    # Step 6: Apply the mask video to the original video
    print("Applying mask video to the original video...")
    apply_mask_to_video(VIDEO_FILE, OUTPUT_VIDEO_FILE, MASKED_VIDEO_FILE)

    print("Processing complete.")


if __name__ == "__main__":
    main()
