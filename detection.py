import cv2
import os
import csv
import argparse
import re
from deepface import DeepFace
import tensorflow as tf
import traceback
import logging
from PIL import Image
import numpy as np

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to convert images to a proper format (RGB, 8-bit)
def preprocess_image(image):
    """Convert image to 8-bit RGB format if necessary."""
    logging.debug(f"[DEBUG] Original image type: {type(image)}, Shape: {image.shape}, dtype: {image.dtype}")
    
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Ensure image is in uint8 format and RGB
    if image.dtype != np.uint8:
        logging.error(f"[ERROR] Unsupported image type: {image.dtype}. Expected uint8.")
        return None

    # Check if the image is in the correct format
    if image.shape[-1] != 3:
        logging.error(f"[ERROR] Image is not in RGB format. Shape: {image.shape}")
        return None

    logging.debug(f"[DEBUG] Processed image type: {type(image)}, Shape: {image.shape}, dtype: {image.dtype}")
    return image  # Already RGB

def collect_training_data(name, student_id, email, section, dataset='dataset', num_samples=30):
    try:
        # Validate student name
        if not re.match(r"^[a-zA-Z\s]+$", name):
            raise ValueError(f"Invalid name format. Only letters and spaces are allowed: {name}")

        # Validate section format
        if not re.match(r"\dCSE[A-Z]", section):
            raise ValueError(f"Invalid section format. Expected format like '4CSEB', but got {section}.")

        # Validate email format
        if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
            raise ValueError(f"Invalid email format: {email}")

        # Create a directory for the student
        student_path = os.path.join(dataset, section, name)
        os.makedirs(student_path, exist_ok=True)

        # Initialize the camera
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise Exception("[ERROR] Camera initialization failed. Please check your camera.")

        count = 1
        logging.info(f"[INFO] Collecting data for {name}. Press ESC to stop early.")

        while count <= num_samples:
            ret, img = cam.read()
            if not ret:
                raise Exception("[ERROR] Error accessing the camera. Exiting...")

            try:
                # Convert frame to RGB with correct format
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Log image shape before processing
                logging.debug(f"[DEBUG] Image shape before preprocessing: {rgb_img.shape}")

                # Preprocess the image
                rgb_img = preprocess_image(rgb_img)

                if rgb_img is None:
                    logging.warning("[WARNING] Skipping image due to unsupported format.")
                    continue

                # Log processed image shape before passing to DeepFace
                logging.debug(f"[DEBUG] Image shape after preprocessing: {rgb_img.shape}")

                # Directly pass the image to DeepFace for face extraction without saving it
                faces = DeepFace.extract_faces(img_path=rgb_img, detector_backend="mtcnn", enforce_detection=False)

                if faces:
                    face = faces[0]["face"]

                    # Ensure face is in uint8 format
                    if face.dtype != "uint8":
                        face = (face * 255).astype("uint8")

                    # Resize face
                    face_resized = cv2.resize(face, (224, 224))
                    if face_resized.dtype != "uint8":
                        face_resized = (face_resized * 255).astype("uint8")

                    # Log face shape before saving
                    logging.debug(f"[DEBUG] Resized face shape: {face_resized.shape}")

                    # Save the detected face image
                    image_path = os.path.join(student_path, f"{count}.jpg")
                    cv2.imwrite(image_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
                    logging.info(f"[SAVED] Image: {image_path}")

                    count += 1
                    cv2.imshow("Face Detection", cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
                else:
                    logging.warning("[WARNING] No face detected. Skipping frame...")

            except Exception as e:
                logging.error(f"[ERROR] Face detection error: {e}")
                traceback.print_exc()

            # Exit early if the user presses ESC
            if cv2.waitKey(10) == 27:  # Check for 'ESC' key
                logging.info("[INFO] Exiting data collection...")
                break

        # Release resources
        cam.release()
        cv2.destroyAllWindows()

        # Save student details to CSV without the path
        csv_file = f"{section}.csv"
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Student ID", "Name", "Email", "Section"])  

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([student_id, name, email, section]) 
            logging.info(f"[INFO] Student data saved to {csv_file}.")

        logging.info(f"[INFO] Data collection for {name} completed. {count - 1} images saved in {student_path}.")

    except Exception as e:
        logging.error(f"[ERROR] An error occurred: {e}")
        traceback.print_exc()
    finally:
        if 'cam' in locals():
            cam.release()
        cv2.destroyAllWindows()

# Main block
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Face Detection Data Collection")
    parser.add_argument("--name", required=True, help="Name of the student")
    parser.add_argument("--student_id", required=True, help="Student ID")
    parser.add_argument("--email", required=True, help="Student email")
    parser.add_argument("--section", required=True, help="Section (e.g., 4CSEB)")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of images to capture")

    args = parser.parse_args()

    # Run the training data collection process
    collect_training_data(
        name=args.name,
        student_id=args.student_id,
        email=args.email,
        section=args.section,
        dataset=args.dataset,
        num_samples=args.num_samples
    )
