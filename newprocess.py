import cv2
import os
from ultralytics import YOLO
from math import radians, cos, sin, sqrt, atan2
import csv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import datetime
import torch

# ------------------------------
# Configuration Section
# ------------------------------

# Define paths
MODEL_PATH = r"Fine_tune.pt"  # Path to your YOLO model
EXTRACTED_FRAMES_DIR = r"extracted_frames50"  # Directory where the extracted frames are stored
OUTPUT_VIDEO_PATH = r"output/VRI_detection_output50.mp4"  # Path to save the final output video
OUTPUT_CSV_PATH = r"output/VRI_detection_data50.csv"  # Path to save detection data

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed

# Video settings
FRAME_WIDTH = 1280  # Adjust based on your frame size
FRAME_HEIGHT = 720
FRAME_RATE = 30

# List of classes (exact class names as used by YOLO)
CLASS_NAMES = [
    "Dirt or Dent",
    "Light Poles",
    "Paint",
    "Scratch",
    "Stickers or Graffiti",
    "VRI"
]

# Distance threshold for assigning unique IDs (in kilometers)
DISTANCE_THRESHOLD = 0.1  # Adjust as needed

# Severity classification thresholds based on object area (number of pixels)
SEVERITY_THRESHOLDS = {
    "A": 1000,  # Lowest severity (smallest area)
    "B": 3000,  # Moderate severity
    "C": 5000,  # Severe
    "D": float('inf')  # Most severe (largest area)
}

# ------------------------------
# Utility Functions
# ------------------------------

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface using the Haversine formula.
    """
    R = 6371.0  # Radius of the Earth in km
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def assign_unique_id(new_vri, detected_vris, class_name, threshold=DISTANCE_THRESHOLD):
    """
    Assign a unique ID to a detected VRI based on geographic proximity within the same class.
    """
    for vri in detected_vris[class_name]:
        distance = calculate_distance(new_vri["lat"], new_vri["lon"], vri["lat"], vri["lon"])
        if distance < threshold:
            return vri["id"]
    new_id = len(detected_vris[class_name]) + 1
    detected_vris[class_name].append({"id": new_id, "lat": new_vri["lat"], "lon": new_vri["lon"]})
    return new_id

def extract_geocoordinates(frame_file, frames_path):
    """
    Extract geographic coordinates (latitude and longitude) from image metadata.
    """
    image_path = os.path.join(frames_path, frame_file)
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data is None:
            raise ValueError("No EXIF data found.")

        gps_info = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "GPSInfo":
                for key in value:
                    sub_tag = GPSTAGS.get(key, key)
                    gps_info[sub_tag] = value[key]

        def get_decimal_from_dms(dms, ref):
            degrees, minutes, seconds = dms
            decimal = degrees + minutes / 60 + seconds / 3600
            if ref in ['S', 'W']:
                decimal = -decimal
            return decimal

        lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
        lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
        return {"lat": lat, "lon": lon}
    except Exception as e:
        print(f"Error extracting geocoordinates from {frame_file}: {e}")
        simulated_lat = 40.0 + (hash(frame_file) % 100) * 0.001
        simulated_lon = -74.0 + (hash(frame_file) % 100) * 0.001
        return {"lat": simulated_lat, "lon": simulated_lon}

def classify_severity(box):
    """
    Classify severity based on the area of the bounding box.
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    area = (x2 - x1) * (y2 - y1)
    for severity, threshold in SEVERITY_THRESHOLDS.items():
        if area <= threshold:
            return severity
    return "D"

# ------------------------------
# Main Processing Function
# ------------------------------

def process_frames_with_yolo(frames_path, output_video_path, model_path, output_csv_path):
    """
    Process image frames using YOLO for object detection, count VRIs, calculate distances, and log detection data.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))

    # Initialize detected VRIs dictionary per class
    detected_vris = {class_name: [] for class_name in CLASS_NAMES}

    # Define fieldnames for CSV using exact class names
    fieldnames = [
        'lat', 'lon', 'Name Image', 'Damage', 'Dirt or Dent', 'Light Poles',
        'Paint', 'Scratch', 'Stickers or Graffiti', 'VRI', 'Grade', 'Number of VRIs', 'Distance Between VRI'
    ]

    # Open CSV file for logging detection data
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through each frame
        for frame_file in sorted(os.listdir(frames_path)):
            frame_path = os.path.join(frames_path, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Failed to load frame: {frame_file}")
                continue

            print(f"Processing frame: {frame_file}")

            # Resize frame to match video dimensions
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Extract or simulate geographic coordinates for the frame
            geo_coords = extract_geocoordinates(frame_file, frames_path)
            frame_lat = geo_coords["lat"]
            frame_lon = geo_coords["lon"]

            # Run object detection
            results = model(frame)

            # Initialize row dictionary for the CSV
            row_data = {
                'lat': frame_lat,
                'lon': frame_lon,
                'Name Image': frame_file.split('.')[0],
                'Damage': '',
                'Dirt or Dent': '',
                'Light Poles': '',
                'Paint': '',
                'Scratch': '',
                'Stickers or Graffiti': '',
                'VRI': '',
                'Grade': '',
                'Number of VRIs': 0,
                'Distance Between VRI': ''
            }

            # Flag to check if damage has been detected
            damage_detected = False
            vri_coords = []  # Store coordinates of detected VRIs

            # Iterate through detection results
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if confidence < CONFIDENCE_THRESHOLD:
                        continue

                    class_name = CLASS_NAMES[class_id]

                    # Only set `Damage` to "Yes" if the class is not "Light Poles"
                    if class_name != "Light Poles":
                        damage_detected = True

                    # Set the corresponding class column to "Yes"
                    if class_name in row_data:
                        severity = ""
                        if class_name == "Paint":
                            severity = classify_severity(box)
                            row_data['Grade'] = severity  # Set severity grade if Paint is detected
                        row_data[class_name] = "Yes"

                    # Record coordinates of detected VRIs for distance calculation
                    if class_name == "VRI":
                        vri_coords.append((frame_lat, frame_lon))

                        # If any class other than "Light Poles" is detected, mark the damage column as "Yes"
            if damage_detected:
                row_data['Damage'] = "Yes"

            # Store the number of VRIs detected in the current frame
            row_data['Number of VRIs'] = len(vri_coords)

            # Calculate distance between VRI objects if multiple VRIs are detected
            if len(vri_coords) > 1:
                distances = [
                    calculate_distance(vri_coords[i][0], vri_coords[i][1], vri_coords[i + 1][0], vri_coords[i + 1][1])
                    for i in range(len(vri_coords) - 1)
                ]
                # Save the sum of distances between detected VRIs in the "Distance Between VRI" column
                row_data['Distance Between VRI'] = round(sum(distances), 4)

            # Write the row to the CSV file
            writer.writerow(row_data)

            # Write the annotated frame to the output video
            video_writer.write(frame)

    # Release resources
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Detection video saved at: {output_video_path}")
    print(f"VRI data saved at: {output_csv_path}")

# ------------------------------
# Script Execution
# ------------------------------

if __name__ == "__main__":
    if not os.path.exists(EXTRACTED_FRAMES_DIR):
        print(f"Extracted frames directory does not exist: {EXTRACTED_FRAMES_DIR}")
        exit(1)

    if not os.path.exists(os.path.dirname(OUTPUT_VIDEO_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
        print(f"Created output video directory: {os.path.dirname(OUTPUT_VIDEO_PATH)}")

    if not os.path.exists(os.path.dirname(OUTPUT_CSV_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        print(f"Created output CSV directory: {os.path.dirname(OUTPUT_CSV_PATH)}")

    process_frames_with_yolo(EXTRACTED_FRAMES_DIR, OUTPUT_VIDEO_PATH, MODEL_PATH, OUTPUT_CSV_PATH)
