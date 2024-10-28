import cv2
import os
from ultralytics import YOLO
from math import radians, cos, sin, sqrt, atan2
import pandas as pd
import json
import pyzed.sl as sl  # ZED SDK for working with SVO files

# ------------------------------
# Configuration Section
# ------------------------------

# Define paths
MODEL_PATH = r"Fine_tune.pt"  # YOLO model path
SVO_FILE_PATH = r"ZED_17.svo2"  # Path to your SVO2 file
GNSS_FILE_PATH = r"gnns/GNSS_17.json"  # Path to your GNSS JSON file
EXTRACTED_FRAMES_DIR = r"extracted_frame1s17"  # Directory where frames will be saved
PROCESSED_FRAMES_DIR = r"output/processed_frames117"  # Directory to save processed frames (with bounding boxes)
OUTPUT_CSV_PATH = r"output/VRI_detection_data.csv"  # Path to save the CSV output
OUTPUT_EXCEL_PATH = r"output/VRI_detection_data.xlsx"  # Path to save the Excel output

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# List of YOLO class names
CLASS_NAMES = [
    "Dirt or Dent",
    "Light Poles",
    "Paint",
    "Scratch",
    "Stickers or Graffiti",
    "VRI"
]

# Function to load GNSS data
def load_gnss_data(gnss_file_path):
    with open(gnss_file_path, 'r') as f:
        gnss_data = json.load(f)["GNSS"]
    return {gnss["ts"]: gnss["coordinates"] for gnss in gnss_data}

# Function to extract only left frames from the SVO file
def extract_frames_from_svo(svo_file_path, frames_dir):
    """
    Extract only the left frames from the SVO2 file and save them into the specified directory.
    """
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file_path)
    init_params.coordinate_units = sl.UNIT.METER
    runtime_params = sl.RuntimeParameters()

    # Attempt to open the SVO file
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Unable to open the SVO file: {status}")
        zed.close()
        return False

    # Prepare to retrieve left frames and depth data
    image = sl.Mat()
    frame_count = 0

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Loop through the SVO file and extract only the left frames
    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)  # Retrieve only the left view

        frame = image.get_data()
        frame_file = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_file, frame)
        print(f"Saved: {frame_file}")
        frame_count += 1

    zed.close()
    print(f"Extracted {frame_count} left frames from SVO.")
    return True

# YOLO Detection and Processing Function
def process_frames_with_yolo(frames_path, processed_frames_dir, model_path, output_excel_path, output_csv_path, gnss_ts_dict):
    """
    Process pre-extracted frames using YOLO, associate detections with GNSS coordinates, and save results to an Excel and CSV file.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # List to store Excel data
    excel_data = []

    # Check if frames exist
    if not os.listdir(frames_path):
        print(f"No frames found in {frames_path}")
        return

    if not os.path.exists(processed_frames_dir):
        os.makedirs(processed_frames_dir)

    # Iterate over each frame
    for frame_file in sorted(os.listdir(frames_path)):
        frame_path = os.path.join(frames_path, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"Failed to load frame: {frame_file}")
            continue

        print(f"Processing frame: {frame_file}")

        # Resize frame (if needed)
        frame = cv2.resize(frame, (1280, 720))

        # Perform YOLO object detection with `save=False` to prevent saving to `runs/predict`
        results = model.predict(source=frame, save=False)
        print(f"Detections in {frame_file}: {results}")

        # Extract frame's timestamp from filename (assuming format like frame_000001.png)
        frame_index = int(frame_file.split('_')[1].split('.')[0])

        # Find the closest GNSS timestamp
        closest_ts = min(gnss_ts_dict.keys(), key=lambda ts: abs(frame_index - ts))  # Matching by index
        coordinates = gnss_ts_dict[closest_ts]
        lat = coordinates["latitude"]
        lon = coordinates["longitude"]

        # Prepare data for Excel
        row_data = {
            'name_image': frame_file.split('.')[0],
            'lat': lat,
            'lon': lon,
            'damaged_asset': '',  # To be filled based on detection
            'Damage_Type': '',
            'Classification': '',  # B, C, D classification
            'Distance_Between_VRIs': ''  # To store distance between multiple VRIs
        }

        # Store the coordinates of VRIs
        vri_coords = []

        # Iterate through detections
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                class_name = CLASS_NAMES[class_id]

                # Set damaged asset and damage type for all classes
                row_data['damaged_asset'] = class_name
                row_data['Damage_Type'] = class_name

                # Add coordinates for VRI detections
                if class_name == "VRI":
                    vri_coords.append((lat, lon))

                # For all classes (except Light Poles), calculate severity
                if class_name != "Light Poles":
                    row_data['Classification'] = classify_severity(box)

                # Draw bounding boxes on the frame
                frame = draw_bounding_boxes(frame, results, CLASS_NAMES)

        # Calculate distances between VRIs (if more than one is detected in the frame)
        if len(vri_coords) > 1:
            distances = []
            for i in range(len(vri_coords) - 1):
                lat1, lon1 = vri_coords[i]
                lat2, lon2 = vri_coords[i + 1]
                distance = calculate_distance(lat1, lon1, lat2, lon2)
                distances.append(distance)
            row_data['Distance_Between_VRIs'] = ', '.join([str(round(d, 4)) for d in distances])  # Store the distances as a comma-separated string

        # Append to Excel data
        excel_data.append(row_data)

        # Save the processed frame with bounding boxes in the output directory
        processed_frame_path = os.path.join(processed_frames_dir, os.path.basename(frame_file))  # Same filename as input
        cv2.imwrite(processed_frame_path, frame)
        print(f"Processed frame saved: {processed_frame_path}")

    # Save to Excel and CSV
    df = pd.DataFrame(excel_data)
    df.to_excel(output_excel_path, index=False)
    df.to_csv(output_csv_path, index=False)
    print(f"Excel and CSV saved to: {output_excel_path} and {output_csv_path}")

# Function to calculate severity based on bounding box area (for classes except Light Poles)
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
    dlon = radians(lat2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Main script to extract frames and process them
if __name__ == "__main__":
    # Load GNSS data
    gnss_ts_dict = load_gnss_data(GNSS_FILE_PATH)

    # Step 1: Extract frames from SVO file (only left frames)
    if extract_frames_from_svo(SVO_FILE_PATH, EXTRACTED_FRAMES_DIR):
        # Step 2: Process the extracted frames with YOLO and GNSS calibration
        process_frames_with_yolo(EXTRACTED_FRAMES_DIR, PROCESSED_FRAMES_DIR, MODEL_PATH, OUTPUT_EXCEL_PATH, OUTPUT_CSV_PATH, gnss_ts_dict)
