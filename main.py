import cv2
import os
import pickle
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from dotenv import load_dotenv
import os

# Load environment variables (for email credentials)
load_dotenv()

# Paths
dataset_path = "dataset"
embeddings_path = "embeddings.pkl"
attendance_file = "attendance.csv"

# Constants
SIMILARITY_THRESHOLD = 0.80  # Increased threshold for better accuracy
already_logged_ids = set()  # Track logged student IDs for the session

# Email credentials from environment variables
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL")  # Use your email here (from .env)
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")  # Use your email password (from .env)
SUBJECT = "Attendance Notification"

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Function to send email
def send_email(recipient_email, student_name):
    try:
        # Create the email message
        message = MIMEMultipart()
        message["From"] = SENDER_EMAIL
        message["To"] = recipient_email
        message["Subject"] = SUBJECT

        body = f"Dear {student_name},\n\nYour attendance has been successfully marked as present for today.\n\nBest regards,\nAttendance System"
        message.attach(MIMEText(body, "plain"))

        # Establish SMTP connection
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Start TLS for security
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, message.as_string())  # Send the email

        logging.info(f"[EMAIL SENT] Attendance notification sent to {student_name} at {recipient_email}.")
    except Exception as e:
        logging.error(f"[EMAIL ERROR] Failed to send email: {e}")

# Function to generate embeddings for all dataset images
def generate_embeddings(dataset_path, output_path):
    embeddings = []
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder '{dataset_path}' not found.")

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                try:
                    embedding = DeepFace.represent(
                        img_path=image_path,
                        model_name="Facenet",
                        enforce_detection=False
                    )
                    if embedding:
                        student_id = os.path.splitext(file)[0]  # Assuming file name is the student ID
                        embeddings.append({
                            "student_id": student_id,
                            "embedding": embedding[0]["embedding"]
                        })
                        logging.info(f"[EMBEDDING GENERATED] {file}")
                except Exception as e:
                    logging.error(f"[ERROR] Failed to generate embedding for {file}: {e}")

    # Save embeddings to file
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    logging.info(f"[INFO] Embeddings saved to '{output_path}'.")

# Load embeddings
def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file '{path}' not found. Run the embedding generation script first.")
    with open(path, "rb") as f:
        return pickle.load(f)

# Face detection before embedding extraction
def detect_faces(frame):
    detected_faces = []
    try:
        faces = DeepFace.extract_faces(frame, detector_backend="opencv", enforce_detection=False)
        for face in faces:
            x, y, w, h = face["facial_area"]["x"], face["facial_area"]["y"], face["facial_area"]["w"], face["facial_area"]["h"]
            detected_faces.append((frame[y:y+h, x:x+w], (x, y, w, h)))  # Crop detected face
    except Exception as e:
        logging.error(f"[FACE DETECTION ERROR] {e}")
    return detected_faces

# Find closest match
def find_closest_match(face_embedding, embeddings):
    face_embedding = face_embedding[0]["embedding"]  # Extract numerical embedding
    similarities = [
        (cosine_similarity([face_embedding], [item["embedding"]])[0][0], item)
        for item in embeddings
    ]
    similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
    if similarities and similarities[0][0] >= SIMILARITY_THRESHOLD:
        return similarities[0][1]
    return None

# Log attendance function (modified to log in class-specific CSV files)
def log_attendance(student_id, student_name=None):
    global already_logged_ids
    if student_id in already_logged_ids:
        logging.debug(f"{student_id} already logged in session.")
        return False

    # Load student details from both sections (4CSEA, 4CSEB)
    student_data = pd.concat([pd.read_csv("4CSEA.csv"), pd.read_csv("4CSEB.csv")], ignore_index=True)
    
    # Strip any leading/trailing whitespaces from the column names
    student_data.columns = student_data.columns.str.strip()

    # Ensure the correct mapping of Name, ID, Email
    if not student_name:
        student_info = student_data[student_data["ID"] == student_id]

        if student_info.empty:
            logging.error(f"Student ID {student_id} not found in database.")
            return False

        # Extracting the name, email, and class
        student_name = student_info.iloc[0]["Name"]
        student_email = student_info.iloc[0]["Email"]
        student_class = student_info.iloc[0]["Class"]

    else:
        student_info = student_data[student_data["ID"] == student_id]
        student_email = student_info.iloc[0]["Email"]
        student_class = student_info.iloc[0]["Class"]

    # Ensure the name is not missing
    if not student_name:
        logging.error(f"Name is missing for student ID {student_id}.")
        return False

    # Update attendance for the current day in the class-specific CSV (e.g., 4CSEA.csv)
    if student_class == "4CSEA":
        csv_file = "4CSEA.csv"
    elif student_class == "4CSEB":
        csv_file = "4CSEB.csv"
    else:
        logging.error(f"Unknown class for student ID {student_id}.")
        return False

    # Load the attendance CSV file for the respective class section
    df = pd.read_csv(csv_file)

    # Strip any leading/trailing whitespaces from column names
    df.columns = df.columns.str.strip()

    # Ensure student ID exists in the CSV file
    student_row = df[df['ID'] == student_id]
    if student_row.empty:
        logging.warning(f"Student with ID {student_id} not found in {csv_file}. Skipping.")
        return False

    # Check and log attendance
    if student_id not in df["ID"].values:
        now = datetime.now()
        new_entry = {
            "Name": student_name,
            "ID": student_id,
            "Email": student_email,
            "Attendance": "Present",
            "Class": student_class,  # Store class instead of section
            "Date": now.strftime("%Y-%m-%d"),
            "Time": now.strftime("%H:%M:%S")
        }
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(csv_file, index=False)
        logging.info(f"[ATTENDANCE LOGGED] {student_name} ({student_id}) in {csv_file}")
        
        # Send email notification
        send_email(student_email, student_name)
        
        already_logged_ids.add(student_id)
        return True
    else:
        logging.info(f"[ALREADY LOGGED] {student_id} already marked present.")
        return False

# Main program
if __name__ == "__main__":
    try:
        # Generate embeddings if not already created
        if not os.path.exists(embeddings_path):
            logging.info("[INFO] Generating embeddings from dataset...")
            generate_embeddings(dataset_path, embeddings_path)

        # Load embeddings
        embeddings = load_embeddings(embeddings_path)
        logging.info("[INFO] Embeddings loaded successfully.")

        # Start camera
        camera = cv2.VideoCapture(0)
        logging.info("\n[INFO] Press 'ESC' to exit.")
        
        while True:
            ret, frame = camera.read()
            if not ret:
                logging.error("[ERROR] Failed to capture frame. Exiting...")
                break

            detected_faces = detect_faces(frame)

            for face, (x, y, w, h) in detected_faces:
                try:
                    face_embedding = DeepFace.represent(
                        img_path=face,
                        model_name="Facenet",
                        enforce_detection=False
                    )
                    if face_embedding:
                        match = find_closest_match(face_embedding, embeddings)
                        if match:
                            log_attendance(match["student_id"], match.get("student_name", match["student_id"]))
                            label = f"ID: {match['student_id']}"  # Display student ID
                            color = (0, 255, 0)  # Green for recognized faces
                            
                        else:
                            label = "Unknown"
                            color = (0, 0, 255)  # Red for unknown faces   
                                           
                        # Draw bounding box around the face
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        # Display the student ID above the bounding box
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                except Exception as e:
                 print(f"[DETECTION ERROR] {e}")

            # Display the image
            cv2.imshow("Face Recognition Attendance System", frame)

            # Exit on 'ESC' key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        camera.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f"[ERROR] An error occurred: {e}")