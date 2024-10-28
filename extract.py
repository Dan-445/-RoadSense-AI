import os
import sys
import argparse
from pyzed import sl
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract images from an SVO file at specified frame intervals.")
    parser.add_argument('--svo_path', type=str, required=True, help="Path to the input SVO file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the extracted images.")
    parser.add_argument('--frame_interval', type=int, required=True, help="Number of frames between each extracted image.")
    return parser.parse_args()

def create_output_directory(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at: {output_dir}")
    else:
        print(f"Output directory already exists at: {output_dir}")

def extract_images(svo_path, output_dir, frame_interval):
    # Initialize the ZED camera
    zed = sl.Camera()

    # Set configuration parameters for SVO file
    input_type = sl.InputType()
    input_type.set_from_svo_file(svo_path)

    init_params = sl.InitParameters(input_t=input_type)
    init_params.coordinate_units = sl.UNIT.METER  # Use meters as units

    # Open the SVO file
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening SVO file: {err}")
        sys.exit(1)

    # Get the camera information
    info = zed.get_camera_information()

    # Retrieve FPS directly from camera_configuration
    fps = info.camera_configuration.fps

    # Handle cases where FPS might not be available
    if fps <= 0:
        fps = 30  # Assume default FPS if unable to retrieve

    # Get total number of frames
    total_frames = zed.get_svo_number_of_frames()

    print(f"SVO FPS: {fps}")
    print(f"Total Frames: {total_frames}")

    print(f"Extracting every {frame_interval} frames")

    # Prepare the image object
    image = sl.Mat()

    frame_idx = 0
    extracted_count = 0

    while True:
        # Grab a new frame
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            if frame_idx % frame_interval == 0:
                # Retrieve left image
                zed.retrieve_image(image, sl.VIEW.LEFT)
                # Convert to numpy array
                img_np = image.get_data()

                # Define the image filename
                image_filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
                # Save the image using OpenCV
                cv2.imwrite(image_filename, img_np)
                print(f"Saved: {image_filename}")
                extracted_count += 1

            frame_idx += 1
        else:
            break  # No more frames

    print(f"Extraction completed. Total images saved: {extracted_count}")

    # Close the camera
    zed.close()

def main():
    args = parse_arguments()

    svo_path = args.svo_path
    output_dir = args.output_dir
    frame_interval = args.frame_interval

    # Validate SVO file path
    if not os.path.isfile(svo_path):
        print(f"SVO file not found at: {svo_path}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    create_output_directory(output_dir)

    # Extract images
    extract_images(svo_path, output_dir, frame_interval)

if __name__ == "__main__":
    main()
