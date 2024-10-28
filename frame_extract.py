import cv2
import os
import pyzed.sl as sl  # ZED SDK for working with SVO files

# Configuration
SVO_FILE_PATH = r"ZED_41.svo2"  # Path to your SVO2 file
EXTRACTED_FRAMES_DIR = r"extracted_frames41"  # Directory where frames will be saved

# Initialize ZED Camera
def extract_frames_from_svo(svo_file_path, frames_dir):
    """
    Extract frames from SVO2 file and save them into the specified directory.
    """
    # Create ZED camera object
    zed = sl.Camera()
    
    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file_path)
    init_params.coordinate_units = sl.UNIT.METER  # Set the unit of measurement
    runtime_params = sl.RuntimeParameters()

    # Open the SVO file
    if not zed.open(init_params) == sl.ERROR_CODE.SUCCESS:
        print(f"Unable to open the SVO file: {svo_file_path}")
        return False

    # Prepare to retrieve frames
    image = sl.Mat()  # To store the frame
    frame_count = 0

    # Create directory if it doesn't exist
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Loop through the SVO and extract frames
    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)  # Retrieve left view frame
        frame = image.get_data()  # Get the frame data
        
        # Define the frame file name and save it
        frame_file = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_file, frame)
        print(f"Saved: {frame_file}")
        frame_count += 1

    zed.close()
    print(f"Extracted {frame_count} frames from the SVO file.")
    return True

# Main execution
if __name__ == "__main__":
    extract_frames_from_svo(SVO_FILE_PATH, EXTRACTED_FRAMES_DIR)
