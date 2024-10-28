from ultralytics import YOLO
import os
import cv2

# Protect main execution from multiprocessing issues
if __name__ == "__main__":
    # Load your pre-trained YOLO model
    model = YOLO('Fine_tune.pt')

    # Define paths
    unlabeled_data_path = 'extracted_frames48'  # Path to your unlabeled images
    pseudo_label_output_path = 'pseudo_labels/'  # Directory to save pseudo-labeled images and annotations

    # Create directories if not exist
    os.makedirs(pseudo_label_output_path, exist_ok=True)

    # Set confidence threshold for pseudo-labeling
    confidence_threshold = 0.6

    # Generate Pseudo-Labels
    for image_file in os.listdir(unlabeled_data_path):
        # Read the image
        image_path = os.path.join(unlabeled_data_path, image_file)

        # Check if the file is an image (skip non-image files)
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {image_file}")
            continue

        # Attempt to read the image using OpenCV
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping corrupt image: {image_file}")
                continue  # Skip to the next image if the image is corrupt
        except Exception as e:
            print(f"Error reading {image_file}: {e}")
            continue  # Skip to the next image in case of error

        # Make predictions using the pre-trained model
        results = model.predict(source=image, save=False, conf=confidence_threshold)

        # Iterate through the results (results is a list of results for each image)
        for result in results:
            # Iterate through each box prediction in the result
            for box in result.boxes:
                # Get confidence score and class of the prediction
                conf = box.conf.item()  # Convert tensor to float
                cls = int(box.cls.item())  # Convert tensor to int
                x_center, y_center, width, height = box.xywh[0].tolist()  # Bounding box coordinates

                # Only consider predictions above the confidence threshold
                if conf >= confidence_threshold:
                    # Save pseudo-label information in a new annotation file
                    annotation_file = os.path.join(pseudo_label_output_path, image_file.replace('.jpg', '.txt'))
                    with open(annotation_file, 'w') as f:
                        f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

                    # Optionally, save the image with bounding boxes drawn for visualization
                    annotated_image = result.plot()  # Get annotated image
                    cv2.imwrite(os.path.join(pseudo_label_output_path, f"annotated_{image_file}"), annotated_image)

    print("Pseudo-labeling completed and annotations saved.")

    # Retrain the model using both the original labeled and pseudo-labeled data
    # Update the data.yaml file to include pseudo-labeled data and labeled data paths
    training_args = {
        'data': 'data.yaml',  # Update the data.yaml file to include pseudo-labeled data
        'epochs': 50,
        'imgsz': 800,
        'batch': 16,
        'lr0': 0.001,
        'optimizer': 'AdamW',
        'patience': 50,
        'save_period': 1,
        'project': 'train',
        'name': 'self_learning_yolov8',  # New experiment name
    }

    # Train the model again with combined data
    try:
        model.train(**training_args)
        print("Training completed successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
