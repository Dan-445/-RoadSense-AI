import cv2
import os
import numpy as np
from ultralytics import YOLO
import pandas as pd
import json
import re
import logging

# ------------------------------
# Configuration Section
# ------------------------------

MODEL_PATH = r"Final_RP_best.pt"  # YOLO model path
GNSS_FILE_PATH = r"gnns/combined_gnss.json"  # Path to your GNSS JSON file
EXTRACTED_FRAMES_DIR = r"extracted_frame43"  # Directory where frames are saved
PROCESSED_FRAMES_DIR = r"43/processed_frames"  # Directory to save processed frames (with bounding boxes)
OUTPUT_CSV_PATH = r"43/VRI_detection_data.csv"  # Path to save the CSV output
OUTPUT_EXCEL_PATH = r"43/VRI_detection_data.xlsx"  # Pacath to save the Excel output
OUTPUT_VIDEO_PATH = r"43/damaged_video.mp4"  # Path to save the MP4 video output

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5

# Frame rate assumption (adjust if different)
FRAME_RATE = 30  # Frames per second

# Only process road-related classes
ROAD_CLASSES = [
    "gesloten verharding-asfaltverharding-dwarsonvlakheid",
    "gesloten verharding-asfaltverharding-rafeling op dicht asfalt",
    "gesloten verharding-asfaltverharding-randschade",
    "gesloten verharding-asfaltverharding-scheurvorming",
    "open verharding-elementenverharding-oneffenheden",
    "open verharding-elementenverharding-ontbrekende/beschadigde elementen",
    "open verharding-elementenverharding-voegwijdte"
]

# YOLO class mapping to ROAD_CLASSES only
YOLO_TO_ROAD_CLASS_MAP = {
    6: "gesloten verharding-asfaltverharding-dwarsonvlakheid",
    7: "gesloten verharding-asfaltverharding-rafeling op dicht asfalt",
    8: "gesloten verharding-asfaltverharding-randschade",
    9: "gesloten verharding-asfaltverharding-scheurvorming",
    10: "open verharding-elementenverharding-oneffenheden",
    11: "open verharding-elementenverharding-ontbrekende/beschadigde elementen",
    12: "open verharding-elementenverharding-voegwijdte"
}

# Scaling factors for coordinate conversion (Adjust based on your calibration)
LAT_PER_PIXEL = 0.00001  # Latitude change per pixel
LON_PER_PIXEL = 0.00001  # Longitude change per pixel

# ------------------------------
# Logging Configuration
# ------------------------------

logging.basicConfig(
    filename='processing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ------------------------------
# Helper Functions
# ------------------------------

def load_gnss_data(gnss_file_path):
    """
    Load GNSS data from a JSON file.
    Assumes each GNSS entry has 'ts', 'coordinates' with 'latitude' and 'longitude'.
    Returns a dictionary mapping timestamp to latitude and longitude.
    """
    try:
        with open(gnss_file_path, 'r') as f:
            gnss_data = json.load(f)["GNSS"]
        gnss_dict = {gnss["ts"]: {"latitude": gnss["coordinates"]["latitude"], "longitude": gnss["coordinates"]["longitude"]} for gnss in gnss_data}
        logging.info(f"Loaded GNSS data from {gnss_file_path}")
        return gnss_dict
    except FileNotFoundError:
        logging.error(f"GNSS file not found at {gnss_file_path}. Exiting.")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding GNSS JSON file: {e}. Exiting.")
        raise
    except KeyError:
        logging.error("Invalid GNSS JSON structure. Expected key 'GNSS'. Exiting.")
        raise

def extract_timestamp_from_filename(filename):
    """
    Extracts the timestamp from a filename (assuming it's part of the filename).
    Example filename: 'frame_1620034801.png'
    """
    timestamp_match = re.search(r'\d+', filename)  # Finds the first occurrence of digits in the filename
    if timestamp_match:
        return int(timestamp_match.group())  # Convert to integer and return
    else:
        logging.warning(f"Timestamp not found in filename: {filename}")
        raise ValueError(f"Timestamp not found in filename: {filename}")

def get_closest_gnss_data(timestamp, gnss_data):
    """
    Finds the closest GNSS data entry for a given timestamp.
    """
    closest_ts = min(gnss_data.keys(), key=lambda ts: abs(ts - timestamp))
    return gnss_data[closest_ts]

def convert_to_gps(bbox, image_size, gnss_lat, gnss_lon):
    """
    Convert bounding box center coordinates to GPS coordinates based on image size and GNSS data.
    bbox: [x_min, y_min, x_max, y_max] -> pixel coordinates of the bounding box.
    image_size: (width, height) of the image.
    gnss_lat, gnss_lon: GNSS latitude and longitude for the frame.
    """
    img_width, img_height = image_size
    # Find the center of the bounding box
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2

    # Calculate the latitude and longitude of the bounding box center
    lat = gnss_lat + ((bbox_center_y - (img_height / 2)) * LAT_PER_PIXEL)
    lon = gnss_lon + ((bbox_center_x - (img_width / 2)) * LON_PER_PIXEL)

    return lat, lon

def classify_severity(box):
    """
    Classify severity based on bounding box area (for road classes).
    Larger bounding boxes correspond to more severe damage.
    Returns B, C, D classification.
    """
    width = box.xyxy[0][2] - box.xyxy[0][0]
    height = box.xyxy[0][3] - box.xyxy[0][1]
    area = width * height / 10000  # Example formula (adjust as needed)

    if area > 100:
        return "B"  # Severe (Very Deep)
    elif 50 < area <= 100:
        return "C"  # Moderate (Deep)
    elif 20 < area <= 50:
        return "D"  # Minor (Moderate or Shallow)
    else:
        return "D"  # Minor (Shallow)

def draw_bounding_box(frame, box, class_name, confidence):
    """
    Draws a bounding box and label on the frame.
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Ensure correct conversion to integers
    label = f"{class_name}: {confidence:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# ------------------------------
# YOLO Detection and Processing Function
# ------------------------------

def process_frames(frames_path, processed_frames_dir, model_path, output_excel_path, output_csv_path, gnss_ts_dict, output_video_path):
    """
    Process extracted frames using YOLO, extract GPS coordinates for road classes only,
    and save results to CSV, Excel, and MP4 video.
    """
    # Load the YOLO model
    try:
        model = YOLO(model_path)
        logging.info(f"YOLO model loaded from {model_path}")
    except Exception as e:
        logging.error(f"Error loading YOLO model: {e}")
        return

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    frame_size = None
    processed_frames_count = 0  # Count of frames written to the video file

    # Ensure processed_frames_dir exists
    if not os.path.exists(processed_frames_dir):
        os.makedirs(processed_frames_dir)
        logging.info(f"Created directory for processed frames: {processed_frames_dir}")

    # Initialize list for Excel and CSV data
    detection_data = []

    # Sort frames based on extracted timestamp
    try:
        sorted_frames = sorted(
            [f for f in os.listdir(frames_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=lambda x: extract_timestamp_from_filename(x)
        )
    except Exception as e:
        logging.error(f"Error sorting frames: {e}")
        return

    for i, frame_file in enumerate(sorted_frames):
        frame_path = os.path.join(frames_path, frame_file)

        try:
            # Extract timestamp from filename
            timestamp = extract_timestamp_from_filename(frame_file)
        except ValueError as ve:
            logging.warning(ve)
            continue

        # Get the closest GNSS data
        try:
            closest_gnss = get_closest_gnss_data(timestamp, gnss_ts_dict)
            gnss_lat = closest_gnss['latitude']
            gnss_lon = closest_gnss['longitude']
        except Exception as e:
            logging.error(f"Error retrieving GNSS data for frame {frame_file}: {e}")
            continue

        # Load the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            logging.warning(f"Failed to load frame: {frame_file}")
            continue

        # Initialize video writer based on first frame's size
        if frame_size is None:
            frame_height, frame_width = frame.shape[:2]
            frame_size = (frame_width, frame_height)
            try:
                video_writer = cv2.VideoWriter(output_video_path, fourcc, FRAME_RATE, frame_size)
                logging.info(f"Video writer initialized: {output_video_path}")
            except Exception as e:
                logging.error(f"Error initializing video writer: {e}")
                return

        logging.info(f"Processing frame: {frame_file} (Timestamp: {timestamp})")

        # Perform YOLO object detection
        try:
            results = model.predict(source=frame, save=False)
        except Exception as e:
            logging.error(f"Error during YOLO prediction for frame {frame_file}: {e}")
            continue

        frame_has_road_detections = False  # Flag to track if any road detections are found

        # Process detections
        for result in results:
            for box in result.boxes:
                try:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                except (IndexError, ValueError) as e:
                    logging.warning(f"Error parsing detection in frame {frame_file}: {e}")
                    continue

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                # Process only road classes
                if class_id in YOLO_TO_ROAD_CLASS_MAP:
                    class_name = YOLO_TO_ROAD_CLASS_MAP[class_id]
                    frame_has_road_detections = True  # Mark that this frame has road class detections

                    # Extract bounding box coordinates
                    bbox = box.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]

                    # Convert bounding box center to GPS coordinates
                    lat, lon = convert_to_gps(bbox, frame_size, gnss_lat, gnss_lon)

                    # Classify severity
                    classification = classify_severity(box)

                    # Prepare data row
                    row = {
                        'name_image': os.path.splitext(frame_file)[0],
                        'lat': lat,
                        'lon': lon,
                        'damaged_asset': "Road",
                        'Damage_Type': class_name,
                        'Classification': classification,
                        'Depth_of_Damage': classification  # Assuming Depth_of_Damage is same as Classification for road classes
                    }

                    detection_data.append(row)

                    # Draw bounding box on the frame
                    draw_bounding_box(frame, box, class_name, confidence)

        # Save the frame only if it has road detections
        if frame_has_road_detections:
            processed_frame_path = os.path.join(processed_frames_dir, frame_file)
            try:
                cv2.imwrite(processed_frame_path, frame)
                logging.info(f"Saved processed frame: {processed_frame_path}")
            except Exception as e:
                logging.error(f"Error saving processed frame {frame_file}: {e}")
                continue

            # Write the frame to the video
            try:
                video_writer.write(frame)
                processed_frames_count += 1
                logging.info(f"Written frame to video: {frame_file}")
            except Exception as e:
                logging.error(f"Error writing frame {frame_file} to video: {e}")
                continue

    # Release the video writer
    if video_writer:
        video_writer.release()
        logging.info(f"Video saved to: {output_video_path}")

    # Save detection data to CSV and Excel
    if detection_data:
        df = pd.DataFrame(detection_data)
        try:
            df.to_csv(output_csv_path, index=False)
            logging.info(f"CSV file saved to: {output_csv_path}")
        except Exception as e:
            logging.error(f"Error saving CSV file: {e}")

        try:
            df.to_excel(output_excel_path, index=False)
            logging.info(f"Excel file saved to: {output_excel_path}")
        except Exception as e:
            logging.error(f"Error saving Excel file: {e}")
    else:
        logging.info("No road class detections were found.")

    logging.info(f"Total frames processed and written to video: {processed_frames_count}")

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    # Load GNSS data
    try:
        gnss_ts_dict = load_gnss_data(GNSS_FILE_PATH)
    except Exception as e:
        logging.error(f"Failed to load GNSS data: {e}")
        exit(1)

    # Process the extracted frames with YOLO and GNSS calibration
    process_frames(
        frames_path=EXTRACTED_FRAMES_DIR,
        processed_frames_dir=PROCESSED_FRAMES_DIR,
        model_path=MODEL_PATH,
        output_excel_path=OUTPUT_EXCEL_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        gnss_ts_dict=gnss_ts_dict,
        output_video_path=OUTPUT_VIDEO_PATH
    )
