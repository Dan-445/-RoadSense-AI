import cv2
import os
import numpy as np
from ultralytics import YOLO
from math import radians, cos, sin, sqrt, atan2, degrees
import pandas as pd
import json
import torch

# ------------------------------
# Configuration Section
# ------------------------------

MODEL_PATH = r"Final_RP_best.pt"  # YOLO model path
GNSS_FILE_PATH = r"gnns/GNSS_41.json"  # Path to your GNSS JSON file
EXTRACTED_FRAMES_DIR = r"extracted_frame"  # Directory where frames are saved
PROCESSED_FRAMES_DIR = r"output/processed_frames"  # Directory to save processed frames (with bounding boxes)
OUTPUT_CSV_PATH = r"output/VRI_detection_data.csv"  # Path to save the CSV output
OUTPUT_EXCEL_PATH = r"output/VRI_detection_data.xlsx"  # Path to save the Excel output
OUTPUT_VIDEO_PATH = r"output/damaged_video.mp4"  # Path to save the MP4 video output

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# List of YOLO class names, categorized as "General" and "Road"
GENERAL_CLASSES = [
    "Dirt or Dent",
    "Light Poles",
    "Paint",
    "Scratch",
    "Stickers or Graffiti",
    "VRI"
]

ROAD_CLASSES = [
    "gesloten verharding-asfaltverharding-dwarsonvlakheid",
    "gesloten verharding-asfaltverharding-rafeling op dicht asfalt",
    "gesloten verharding-asfaltverharding-randschade",
    "gesloten verharding-asfaltverharding-scheurvorming",
    "open verharding-elementenverharding-oneffenheden",
    "open verharding-elementenverharding-ontbrekende/beschadigde elementen",
    "open verharding-elementenverharding-voegwijdte"
]

# Function to load GNSS data
def load_gnss_data(gnss_file_path):
    with open(gnss_file_path, 'r') as f:
        gnss_data = json.load(f)["GNSS"]
    return {gnss["ts"]: gnss["coordinates"] for gnss in gnss_data}

# Function to interpolate latitude and longitude based on object position in frame
def interpolate_gnss(lat_start, lon_start, lat_end, lon_end, x_pos, y_pos, frame_width, frame_height):
    """
    Interpolate GNSS coordinates based on object detection position within the frame.
    This assumes that the object positions within the frame contribute to latitude and longitude interpolation.
    """
    lat = lat_start + (lat_end - lat_start) * (x_pos / frame_width)
    lon = lon_start + (lon_end - lon_start) * (y_pos / frame_height)
    return lat, lon

# Function to calculate depth of damage for road-related classes using B, C, D classification
def calculate_depth_of_damage(box):
    """
    Calculate the depth of damage based on the bounding box.
    Larger bounding boxes correspond to deeper damage.
    Return B, C, D classification for road-related classes.
    """
    width = box.xyxy[0][2] - box.xyxy[0][0]
    height = box.xyxy[0][3] - box.xyxy[0][1]
    depth = width * height / 10000  # Example formula (you can modify it)

    if depth > 100:
        return "B"  # Severe (Very Deep)
    elif 50 < depth <= 100:
        return "C"  # Moderate (Deep)
    elif 20 < depth <= 50:
        return "D"  # Minor (Moderate or Shallow)
    else:
        return "D"  # Minor (Shallow)

# Function to calculate distance from camera to object using the depth map
def calculate_distance_from_camera(depth_map, box):
    """
    Calculate the distance from the camera to the object using the depth map.
    The distance is calculated at the center of the bounding box.
    """
    x_center = int((box[0] + box[2]) / 2)
    y_center = int((box[1] + box[3]) / 2)
    distance = depth_map[y_center, x_center]  # Depth in meters at the center of the object
    return distance

# YOLO Detection and Processing Function with Distance Calculation
def process_frames_with_yolo(frames_path, processed_frames_dir, model_path, output_excel_path, output_csv_path, gnss_ts_dict, output_video_path):
    """
    Process pre-extracted frames using YOLO, associate detections with GNSS coordinates, calculate distance to objects, and save results to an Excel, CSV, and MP4 video.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # List to store Excel data
    excel_data = []
    frame_size = (1280, 720)
    
    # Initialize video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, frame_size)
        print(f"Video writer initialized: {output_video_path}")
    except Exception as e:
        print(f"Error initializing video writer: {e}")
        return

    # Check if frames exist
    if not os.listdir(frames_path):
        print(f"No frames found in {frames_path}")
        return

    if not os.path.exists(processed_frames_dir):
        os.makedirs(processed_frames_dir)

    processed_frames_count = 0  # Count of frames written to the video file

    # Iterate over each frame
    for frame_file in sorted(os.listdir(frames_path)):
        if frame_file.endswith(".png"):  # Ensure you're only processing the frames, not the depth files
            frame_path = os.path.join(frames_path, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Failed to load frame: {frame_file}")
                continue

            print(f"Processing frame: {frame_file}")

            # Load the depth map associated with this frame
            depth_file = frame_path.replace(".png", ".npy")  # Assume corresponding depth file is saved as .npy
            depth_map = np.load(depth_file)

            # Resize frame (if needed)
            frame = cv2.resize(frame, frame_size)

            # Perform YOLO object detection
            results = model.predict(source=frame, save=False)

            # Extract frame's timestamp from filename (assuming format like frame_000001.png)
            frame_index = int(frame_file.split('_')[1].split('.')[0])

            # Find the closest GNSS timestamps (for interpolation)
            gnss_keys = sorted(gnss_ts_dict.keys())  # Ensure GNSS timestamps are sorted

            try:
                if frame_index < min(gnss_keys):
                    closest_ts_start = min(gnss_keys)
                    closest_ts_end = min(gnss_keys)
                elif frame_index > max(gnss_keys):
                    closest_ts_start = max(gnss_keys)
                    closest_ts_end = max(gnss_keys)
                else:
                    closest_ts_start = max(k for k in gnss_keys if k <= frame_index)
                    closest_ts_end = min(k for k in gnss_keys if k >= frame_index)
            except ValueError:
                print(f"Warning: No matching GNSS timestamp found for frame {frame_file} (frame_index={frame_index}). Skipping this frame.")
                continue

            lat_start, lon_start = gnss_ts_dict[closest_ts_start]["latitude"], gnss_ts_dict[closest_ts_start]["longitude"]
            lat_end, lon_end = gnss_ts_dict[closest_ts_end]["latitude"], gnss_ts_dict[closest_ts_end]["longitude"]

            # Prepare data for Excel
            row_data = {
                'name_image': frame_file.split('.')[0],
                'lat': None,  # To be filled based on object detection position
                'lon': None,  # To be filled based on object detection position
                'damaged_asset': '',  # To be filled based on detection
                'Damage_Type': '',
                'Classification': '',  # B, C, D classification for general and road classes
                'Depth_of_Damage': '',  # Depth of damage for road classes
                'Distance_Between_VRIs': '',  # To store distance between multiple VRIs
                'Number_of_VRIs': 0,  # To store number of VRIs detected in the frame
                'distance_from_camera': None  # Distance from camera in meters
            }

            vri_coords = []  # Store coordinates of VRIs
            detections = {}  # To track unique detections (to eliminate duplicates)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    class_name = GENERAL_CLASSES[class_id] if class_id < len(GENERAL_CLASSES) else ROAD_CLASSES[class_id - len(GENERAL_CLASSES)]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Interpolate lat/lon based on detection location in the frame
                    lat, lon = interpolate_gnss(lat_start, lon_start, lat_end, lon_end, x1, y1, frame_size[0], frame_size[1])
                    row_data['lat'] = lat
                    row_data['lon'] = lon

                    # Calculate distance from the camera to the detected object using the depth map
                    distance_from_camera = calculate_distance_from_camera(depth_map, [x1, y1, x2, y2])
                    row_data['distance_from_camera'] = distance_from_camera

                    # Only store unique detections based on class name and coordinates
                    detection_key = (class_name, lat, lon)
                    if detection_key not in detections:
                        detections[detection_key] = True

                        if class_name in GENERAL_CLASSES:
                            # Set damaged asset and damage type for all general classes
                            row_data['damaged_asset'] = class_name
                            row_data['Damage_Type'] = class_name

                            # Severity classification for all classes except "Light Poles"
                            if class_name != "Light Poles":
                                row_data['Classification'] = classify_severity(box)

                        elif class_name in ROAD_CLASSES:
                            # Handle road-related classes under the "road" column and classify depth of damage
                            row_data['damaged_asset'] = "Road"  # Set damaged_asset to "Road"
                            row_data['Damage_Type'] = class_name  # Set Damage_Type to the detected road class
                            row_data['Classification'] = calculate_depth_of_damage(box)  # Classify depth of damage as B, C, D

                        if class_name == "VRI":
                            vri_coords.append((lat, lon))
                            row_data['Number_of_VRIs'] += 1  # Increment VRI count

                        # Draw bounding boxes on the frame
                        frame = draw_bounding_boxes(frame, results, GENERAL_CLASSES + ROAD_CLASSES)

            if detections:  # Only save frames with damages
                # Calculate distances between VRIs if more than one detected
                if len(vri_coords) > 1:
                    distances = []
                    for i in range(len(vri_coords) - 1):
                        lat1, lon1 = vri_coords[i]
                        lat2, lon2 = vri_coords[i + 1]
                        distance = calculate_distance(lat1, lon1, lat2, lon2)
                        distances.append(distance)
                    row_data['Distance_Between_VRIs'] = ', '.join([str(round(d, 4)) for d in distances])

                # Append to Excel data
                excel_data.append(row_data)

                # Save the processed frame with bounding boxes in the output directory
                processed_frame_path = os.path.join(processed_frames_dir, os.path.basename(frame_file))
                cv2.imwrite(processed_frame_path, frame)

                # Write frame to the video file
                video_writer.write(frame)
                processed_frames_count += 1  # Increment count of processed frames

    # Release the video writer
    video_writer.release()
    print(f"Total frames written to video: {processed_frames_count}")

    # Save to Excel and CSV
    df = pd.DataFrame(excel_data)
    df.to_excel(output_excel_path, index=False)
    df.to_csv(output_csv_path, index=False)
    print(f"Excel, CSV, and video saved to: {output_excel_path}, {output_csv_path}, and {output_video_path}")

# Function to calculate severity based on bounding box area (for general classes only)
def classify_severity(box):
    area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
    if area > 50000:
        return "B"  # Severe
    elif 20000 < area <= 50000:
        return "C"  # Moderate
    else:
        return "D"  # Minor

# Function to draw bounding boxes on frames
def draw_bounding_boxes(frame, results, class_names):
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_names[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Helper Function to calculate distance between VRIs
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Main script to process the extracted frames
if __name__ == "__main__":
    # Load GNSS data
    gnss_ts_dict = load_gnss_data(GNSS_FILE_PATH)

    # Process the extracted frames with YOLO and GNSS calibration
    process_frames_with_yolo(EXTRACTED_FRAMES_DIR, PROCESSED_FRAMES_DIR, MODEL_PATH, OUTPUT_EXCEL_PATH, OUTPUT_CSV_PATH, gnss_ts_dict, OUTPUT_VIDEO_PATH)
