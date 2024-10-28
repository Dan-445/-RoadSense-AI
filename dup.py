import cv2
import os
import numpy as np
from ultralytics import YOLO
import pandas as pd
import json

# ------------------------------
# Configuration Section
# ------------------------------

MODEL_PATH = r"Final_RP_best.pt"  # YOLO model path
GNSS_FILE_PATH = r"gnns/GNSS_41.json"  # Path to your GNSS JSON file
EXTRACTED_FRAMES_DIR = r"extracted_frame41"  # Directory where frames are saved
PROCESSED_FRAMES_DIR = r"output/processed_frames"  # Directory to save processed frames (with bounding boxes)
OUTPUT_CSV_PATH = r"output/VRI_detection_data.csv"  # Path to save the CSV output
OUTPUT_EXCEL_PATH = r"output/VRI_detection_data.xlsx"  # Path to save the Excel output
OUTPUT_VIDEO_PATH = r"output/damaged_video.mp4"  # Path to save the MP4 video output

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# Function to check if the directory exists and create it if it doesn't
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# YOLO Detection and Processing Function with Saving Only Detected Frames and Results
def process_frames_with_yolo(frames_path, processed_frames_dir, model_path, output_excel_path, output_csv_path, output_video_path):
    model = YOLO(model_path)
    excel_data = []
    
    # Get the frame size dynamically from the first image
    first_frame_file = sorted([file for file in os.listdir(frames_path) if file.endswith(".png")])[0]
    first_frame_path = os.path.join(frames_path, first_frame_file)
    
    first_frame = cv2.imread(first_frame_path, cv2.IMREAD_UNCHANGED)
    if first_frame is None:
        print(f"Failed to load first frame: {first_frame_path}")
        return

    frame_size = (first_frame.shape[1], first_frame.shape[0])  # (width, height) based on the first frame
    print(f"Detected frame size: {frame_size}")

    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, frame_size)
        print(f"Video writer initialized with frame size {frame_size}: {output_video_path}")
    except Exception as e:
        print(f"Error initializing video writer: {e}")
        return

    ensure_directory_exists(processed_frames_dir)
    processed_frames_count = 0

    for frame_file in sorted(os.listdir(frames_path)):
        if frame_file.endswith(".png"):
            frame_path = os.path.join(frames_path, frame_file)
            print(f"Loading frame: {frame_path}")
            
            frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                print(f"Failed to load frame with cv2.imread(): {frame_file}")
                continue

            # Check if the image has an alpha channel (4 channels - RGBA)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                print(f"Converted RGBA to RGB for frame: {frame_file}")

            # Perform YOLO inference
            results = model.predict(source=frame, save=False)
            print(f"YOLO results for frame {frame_file}: {results}")

            # Check if there are any detections in the frame
            detections_in_frame = False
            for result in results:
                for box in result.boxes:
                    if float(box.conf[0]) >= CONFIDENCE_THRESHOLD:
                        detections_in_frame = True
                        break

            # If no detections, skip saving and writing this frame
            if not detections_in_frame:
                print(f"No detections in frame {frame_file}, skipping...")
                continue

            # Save the frame with detections
            processed_frame_path = os.path.join(processed_frames_dir, os.path.basename(frame_file))
            try:
                cv2.imwrite(processed_frame_path, frame)
                print(f"Processed frame with detection saved: {processed_frame_path}")
            except Exception as e:
                print(f"Error saving processed frame: {processed_frame_path}. Error: {e}")

            # Write the frame with detections to the video
            try:
                video_writer.write(frame)
                print(f"Frame with detection written to video: {frame_file}")
            except Exception as e:
                print(f"Error writing frame to video: {frame_file}. Error: {e}")

            # Example data for Excel/CSV row (modify according to your needs)
            row_data = {
                'name_image': frame_file.split('.')[0],
                'lat': None,  # Placeholder: replace with actual latitude
                'lon': None,  # Placeholder: replace with actual longitude
                'damaged_asset': '',  # Placeholder: replace with actual damaged asset
                'damage_type': '',  # Placeholder: replace with actual damage type
                'classification': '',  # Placeholder: replace with actual classification
                'depth_of_damage': '',  # Placeholder: replace with actual depth of damage
                'distance_between_vris': '',  # Placeholder: replace with actual distance between VRIs
                'number_of_vris': 0,  # Placeholder: replace with actual number of VRIs
                'distance_from_camera': None  # Placeholder: replace with actual distance from camera
            }

            excel_data.append(row_data)
            processed_frames_count += 1

    video_writer.release()
    print(f"Total frames with detections written to video: {processed_frames_count}")

    # Save Excel/CSV with detected frames only
    df = pd.DataFrame(excel_data)
    try:
        df.to_excel(output_excel_path, index=False)
        print(f"Excel saved to: {output_excel_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
    
    try:
        df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

# Main script to process frames
if __name__ == "__main__":
    process_frames_with_yolo(EXTRACTED_FRAMES_DIR, PROCESSED_FRAMES_DIR, MODEL_PATH, OUTPUT_EXCEL_PATH, OUTPUT_CSV_PATH, OUTPUT_VIDEO_PATH)
