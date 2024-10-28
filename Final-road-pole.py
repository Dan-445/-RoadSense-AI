import cv2
import os
import numpy as np
from ultralytics import YOLO
from math import radians, cos, sin, sqrt, atan2, degrees
import pandas as pd
import json
import pyzed.sl as sl  # ZED SDK for working with SVO files

# ------------------------------
# Configuration Section
# ------------------------------

MODEL_PATH = r"Final_RP_best.pt"  # YOLO model path
SVO_FILE_PATH = r"ZED_41.svo2"  # Path to your SVO2 file
GNSS_FILE_PATH = r"gnns/GNSS_41.json"  # Path to your GNSS JSON file
EXTRACTED_FRAMES_DIR = r"extracted_frame41"  # Directory where frames will be saved
PROCESSED_FRAMES_DIR = r"output41/processed_frames"  # Directory to save processed frames (with bounding boxes)
OUTPUT_CSV_PATH = r"output41/VRI_detection_data.csv"  # Path to save the CSV output
OUTPUT_EXCEL_PATH = r"output41/VRI_detection_data.xlsx"  # Path to save the Excel output
OUTPUT_VIDEO_PATH = r"output41/damaged_video.mp4"  # Path to save the MP4 video output

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

# Function to extract only left frames from the SVO file and save depth maps as .npy files
def extract_frames_from_svo(svo_file_path, frames_dir):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file_path)
    init_params.coordinate_units = sl.UNIT.METER
    runtime_params = sl.RuntimeParameters()

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Unable to open the SVO file: {status}")
        zed.close()
        return False

    image = sl.Mat()
    depth = sl.Mat()
    frame_count = 0

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        frame = image.get_data()
        depth_data = depth.get_data()

        frame_file = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
        depth_file = os.path.join(frames_dir, f"frame_{frame_count:06d}.npy")
        
        cv2.imwrite(frame_file, frame)
        np.save(depth_file, depth_data)
        print(f"Saved: {frame_file}, Depth saved: {depth_file}")
        
        frame_count += 1

    zed.close()
    print(f"Extracted {frame_count} left frames and depth data from SVO.")
    return True

# Function to calculate the new latitude and longitude using distance and bearing
def calculate_new_lat_lon(lat1, lon1, distance, bearing):
    # Ensure the distance is valid (non-negative and not too large)
    if distance <= 0 or distance > 40000:  # Earth circumference in kilometers ~40,000 km
        print(f"Warning: Invalid distance {distance}. Skipping this calculation.")
        return lat1, lon1  # Return the original latitude and longitude if distance is invalid
    
    R = 6371.0  # Earth's radius in kilometers
    lat1 = radians(lat1)
    lon1 = radians(lon1)

    lat2 = sin(lat1) * cos(distance / R) + cos(lat1) * sin(distance / R) * cos(radians(bearing))
    lon2 = lon1 + atan2(sin(radians(bearing)) * sin(distance / R) * cos(lat1),
                        cos(distance / R) - sin(lat1) * sin(lat2))

    return (degrees(lat2), degrees(lon2))

# Function to calculate the bearing from two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    
    y = sin(dLon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
    bearing = atan2(y, x)
    return (degrees(bearing) + 360) % 360

# Function to calculate distance from camera to object using the depth map
def calculate_distance_from_camera(depth_map, box):
    x_center = int((box[0] + box[2]) / 2)
    y_center = int((box[1] + box[3]) / 2)
    
    # Fetch the depth value at the center of the bounding box
    distance = depth_map[y_center, x_center]

    # Check for invalid depth values (negative or NaN)
    if np.isnan(distance) or distance < 0:
        print(f"Warning: Invalid depth value {distance} at coordinates ({x_center}, {y_center}).")
        return 0  # Return a default value if distance is invalid

    return distance

# Function to calculate depth of damage for road-related classes using B, C, D classification
def calculate_depth_of_damage(box):
    # box.xyxy contains [x1, y1, x2, y2], which represent the bounding box coordinates
    x1, y1, x2, y2 = box.xyxy[0]  # Extract coordinates from box.xyxy

    width = x2 - x1
    height = y2 - y1
    area = width * height

    if area > 10000:
        return "B"  # Severe (Very Deep)
    elif 5000 < area <= 10000:
        return "C"  # Moderate (Deep)
    else:
        return "D"  # Minor (Shallow)

# YOLO Detection and Processing Function with Distance Calculation
def process_frames_with_yolo(frames_path, processed_frames_dir, model_path, output_excel_path, output_csv_path, gnss_ts_dict, output_video_path):
    model = YOLO(model_path)
    excel_data = []
    frame_size = (1280, 720)
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, frame_size)
        print(f"Video writer initialized: {output_video_path}")
    except Exception as e:
        print(f"Error initializing video writer: {e}")
        return

    if not os.path.exists(processed_frames_dir):
        os.makedirs(processed_frames_dir)

    processed_frames_count = 0

    for frame_file in sorted(os.listdir(frames_path)):
        if frame_file.endswith(".png"):
            frame_path = os.path.join(frames_path, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Failed to load frame: {frame_file}")
                continue

            print(f"Processing frame: {frame_file}")

            depth_file = frame_path.replace(".png", ".npy")
            depth_map = np.load(depth_file)

            # Ensure frame size matches the expected size for video writer
            if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
                frame = cv2.resize(frame, frame_size)  # Resize to match video writer frame size

            results = model.predict(source=frame, save=False)

            frame_index = int(frame_file.split('_')[1].split('.')[0])

            gnss_keys = sorted(gnss_ts_dict.keys())

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

            row_data = {
                'name_image': frame_file.split('.')[0],
                'lat': None,
                'lon': None,
                'damaged_asset': '',
                'Damage_Type': '',
                'Classification': '',
                'Depth_of_Damage': '',
                'Distance_Between_VRIs': '',
                'Number_of_VRIs': 0,
                'distance_from_camera': None
            }

            vri_coords = []
            detections = {}

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    class_name = GENERAL_CLASSES[class_id] if class_id < len(GENERAL_CLASSES) else ROAD_CLASSES[class_id - len(GENERAL_CLASSES)]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Use the distance from the depth map and calculate the bearing
                    distance_from_camera = calculate_distance_from_camera(depth_map, [x1, y1, x2, y2])
                    row_data['distance_from_camera'] = distance_from_camera
                    
                    # Calculate the bearing from the frame GNSS start and end points
                    bearing = calculate_bearing(lat_start, lon_start, lat_end, lon_end)
                    
                    # Calculate new latitude and longitude using the distance and bearing
                    new_lat, new_lon = calculate_new_lat_lon(lat_start, lon_start, distance_from_camera, bearing)
                    row_data['lat'] = new_lat
                    row_data['lon'] = new_lon

                    detection_key = (class_name, new_lat, new_lon)
                    if detection_key not in detections:
                        detections[detection_key] = True

                        if class_name in GENERAL_CLASSES:
                            row_data['damaged_asset'] = class_name
                            row_data['Damage_Type'] = class_name

                            if class_name != "Light Poles":
                                row_data['Classification'] = classify_severity(box)

                        elif class_name in ROAD_CLASSES:
                            row_data['damaged_asset'] = "Road"
                            row_data['Damage_Type'] = class_name
                            row_data['Classification'] = calculate_depth_of_damage(box)

                        if class_name == "VRI":
                            vri_coords.append((new_lat, new_lon))
                            row_data['Number_of_VRIs'] += 1

                        frame = draw_bounding_boxes(frame, results, GENERAL_CLASSES + ROAD_CLASSES)

            if detections:
                if len(vri_coords) > 1:
                    distances = []
                    for i in range(len(vri_coords) - 1):
                        lat1, lon1 = vri_coords[i]
                        lat2, lon2 = vri_coords[i + 1]
                        distance = calculate_distance(lat1, lon1, lat2, lon2)
                        distances.append(distance)
                    row_data['Distance_Between_VRIs'] = ', '.join([str(round(d, 4)) for d in distances])

                excel_data.append(row_data)

                processed_frame_path = os.path.join(processed_frames_dir, os.path.basename(frame_file))
                cv2.imwrite(processed_frame_path, frame)

                video_writer.write(frame)
                processed_frames_count += 1

    video_writer.release()
    print(f"Total frames written to video: {processed_frames_count}")

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

# Main script to extract frames and process them
if __name__ == "__main__":
    gnss_ts_dict = load_gnss_data(GNSS_FILE_PATH)

    if extract_frames_from_svo(SVO_FILE_PATH, EXTRACTED_FRAMES_DIR):
        process_frames_with_yolo(EXTRACTED_FRAMES_DIR, PROCESSED_FRAMES_DIR, MODEL_PATH, OUTPUT_EXCEL_PATH, OUTPUT_CSV_PATH, gnss_ts_dict, OUTPUT_VIDEO_PATH)
