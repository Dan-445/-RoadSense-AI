import os
from PIL import Image, ImageFilter

# ------------------------------
# Configuration Section
# ------------------------------

SOURCE_FOLDER = "dataset"  # Folder containing the images
TARGET_FOLDER = "dataset_f"  # Folder to save resized images
MINIMUM_SIZE_MB = 6  # Minimum size in MB
TARGET_RESOLUTION = (7680, 4320)  # 8K resolution (higher than 4K)
MAX_QUALITY = 100  # Maximum quality (JPEG quality range is 0-100)
IMAGE_FORMAT = "JPEG"  # You can change this to 'PNG' if you want higher file sizes

# ------------------------------
# Helper Functions
# ------------------------------

def resize_image(image, target_resolution):
    """
    Resizes the image to the target resolution (e.g., 8K).
    """
    return image.resize(target_resolution, Image.Resampling.LANCZOS)

def save_image_with_minimum_size(image, output_path, min_size_mb, image_format):
    """
    Saves the image with an iterative quality adjustment to ensure
    it meets the minimum file size requirement.
    """
    quality = 90  # Start with high quality
    while quality <= MAX_QUALITY:
        # Save the image with the current quality setting
        image.save(output_path, format=image_format, quality=quality)

        # Check the file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)  # Convert bytes to MB
        if file_size_mb >= min_size_mb:
            print(f"Image saved with quality={quality} and size={file_size_mb:.2f} MB")
            break  # Exit the loop when the file size meets or exceeds the requirement
        else:
            print(f"Current file size is {file_size_mb:.2f} MB; increasing quality...")

        # Increase the quality for the next iteration
        quality += 2

def add_noise(image):
    """
    Adds a subtle noise to the image to increase its size.
    """
    return image.filter(ImageFilter.GaussianBlur(0.5))  # Adds slight noise

# ------------------------------
# Main Processing Function
# ------------------------------

def process_images(source_folder, target_folder, min_size_mb, target_resolution, image_format):
    """
    Process all images in the source folder, resize them to 8K resolution,
    add noise, and adjust their quality to meet the minimum size requirement.
    """
    # Ensure the target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Process each image in the source folder
    for image_name in os.listdir(source_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(source_folder, image_name)
            output_path = os.path.join(target_folder, image_name)

            try:
                # Open the image
                with Image.open(image_path) as img:
                    # Resize the image to 8K resolution
                    resized_img = resize_image(img, target_resolution)

                    # Add noise to the image to increase the file size
                    noisy_img = add_noise(resized_img)

                    # Save the image with quality adjustments to meet the minimum size requirement
                    save_image_with_minimum_size(noisy_img, output_path, min_size_mb, image_format)
                    print(f"Processed image: {image_name}")
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    process_images(
        source_folder=SOURCE_FOLDER,
        target_folder=TARGET_FOLDER,
        min_size_mb=MINIMUM_SIZE_MB,
        target_resolution=TARGET_RESOLUTION,
        image_format=IMAGE_FORMAT
    )
