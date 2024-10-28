import os

EXTRACTED_FRAMES_DIR = r"extracted_frame41"  # Directory where frames are saved

# Verify if directory exists
if not os.path.exists(EXTRACTED_FRAMES_DIR):
    print(f"Directory does not exist: {EXTRACTED_FRAMES_DIR}")
else:
    print(f"Directory exists: {EXTRACTED_FRAMES_DIR}")

# Check if there are PNG and NPY files in the directory
files_in_directory = os.listdir(EXTRACTED_FRAMES_DIR)
png_files = [file for file in files_in_directory if file.endswith(".png")]
npy_files = [file for file in files_in_directory if file.endswith(".npy")]

# Debug file detection
print(f"Found {len(png_files)} PNG files.")
print(f"Found {len(npy_files)} NPY files.")

# Check if PNG and NPY files are properly matched
for png_file in png_files:
    npy_file = png_file.replace(".png", ".npy")
    if npy_file not in npy_files:
        print(f"Missing corresponding NPY file for: {png_file}")
    else:
        print(f"Matched: {png_file} and {npy_file}")

# If everything is matched correctly, proceed with your processing
if len(png_files) > 0 and len(npy_files) > 0:
    print("Proceeding with frame processing...")
    # Here you can call your frame processing function
else:
    print("No valid frames or NPY files to process.")
