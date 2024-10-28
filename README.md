
# RoadSense-AI

![RoadSense-AI Logo](assets/logo.png) <!-- Replace with actual logo if available -->

**RoadSense-AI** is an advanced AI-driven system designed to detect and analyze various road damages using SVO2 camera technology. By leveraging the power of YOLO (You Only Look Once) object detection and GNSS (Global Navigation Satellite System) data, RoadSense-AI provides comprehensive insights into road conditions, facilitating efficient maintenance and infrastructure management.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Advanced Object Detection:** Utilizes YOLO for real-time detection of road-related damages such as dirt, dents, scratches, graffiti, and more.
- **Depth Analysis:** Integrates depth maps to calculate the severity and depth of detected damages.
- **GNSS Integration:** Associates detections with precise geographical coordinates for accurate location mapping.
- **Comprehensive Reporting:** Generates detailed reports in CSV and Excel formats, including distance calculations between multiple VRIs (Virtual Road Infrastructures).
- **Video Processing:** Produces annotated videos highlighting detected damages with bounding boxes and labels.
- **Scalable Architecture:** Easily extendable to include additional classes and functionalities as needed.

## Demo

![RoadSense-AI Demo](assets/demo.gif) <!-- Replace with actual demo GIF or images if available -->

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Operating System:** Windows, macOS, or Linux
- **Python Version:** Python 3.7 or higher
- **Hardware:** Compatible with ZED SDK and capable of handling video processing tasks
- **Software Dependencies:**
  - [Git](https://git-scm.com/)
  - [Python](https://www.python.org/downloads/)
  - [ZED SDK](https://www.stereolabs.com/developers/)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Dan-445/RoadSense-AI.git
   cd RoadSense-AI
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Python Packages**

   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt`, you can install the dependencies manually:

   ```bash
   pip install opencv-python numpy ultralytics pandas pyzed
   ```

4. **Install ZED SDK**

   - Download and install the ZED SDK from [here](https://www.stereolabs.com/developers/).
   - Follow the installation instructions specific to your operating system.

## Configuration

### Model Path
Ensure you have the YOLO model (`Fine_tune.pt`) placed in the root directory or update the `MODEL_PATH` in the script to point to its location.

### SVO2 File
Place your SVO2 file (e.g., `ZED_43.svo2`) in the designated directory or update the `SVO_FILE_PATH` accordingly.

### GNSS Data
Ensure your GNSS JSON file (e.g., `GNSS_43.json`) is correctly formatted and placed in the `gnss` directory or update the `GNSS_FILE_PATH`.

### Directories

- Extracted Frames: `extracted_frame434`
- Processed Frames: `output/processed_frames434`
- Output Files: `output/`

Make sure these directories exist or let the script create them automatically.

### Confidence Threshold
Adjust the `CONFIDENCE_THRESHOLD` in the script if needed to filter detections based on confidence levels.

## Usage

### Prepare Your Data
- **SVO2 File:** Ensure your SVO2 file contains the video data you wish to process.
- **GNSS JSON File:** Ensure your GNSS data is correctly formatted and synchronized with the video frames.

### Run the Script
Execute the main script to start processing:

```bash
python main.py
```

Ensure that `main.py` contains the provided code or adjust accordingly.

### Monitoring Progress
The script will:
- Extract frames and depth maps from the SVO2 file.
- Perform YOLO-based object detection on each frame.
- Associate detections with GNSS coordinates.
- Calculate distances and classify the severity of damages.
- Generate annotated frames, a compiled video, and detailed reports in CSV and Excel formats.

## Output

After successful execution, the following outputs will be generated:

- **Processed Frames:** Annotated images with bounding boxes and labels stored in `output/processed_frames434`.
- **Video Output:** A compiled video (`43damaged_video.mp4`) highlighting all detections.
- **Reports:**
  - CSV: `43VRI_detection_data.csv`
  - Excel: `43VRI_detection_data.xlsx`

These files provide a comprehensive overview of detected road damages, their locations, severity classifications, and distances between multiple VRIs.


Happy coding and best of luck with RoadSense-AI!
