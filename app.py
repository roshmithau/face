import cv2
import os
import pickle
import csv
import base64
from flask import Flask, render_template, request, redirect, url_for
from deepface import DeepFace
from io import BytesIO
from PIL import Image
from datetime import datetime
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess

app = Flask(__name__)

from flask import Flask

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Set to 32MB


# Paths
dataset_path = "dataset"
embeddings_path = "embeddings.pkl"
students_file = "students.csv"
attendance_file = "attendance.csv"

# Email Configurations
email_sender = "your_email@gmail.com"  # Replace with your email
email_password = "your_email_password"  # Replace with your email password
smtp_server = "smtp.gmail.com"
smtp_port = 587

# Constants
SIMILARITY_THRESHOLD = 0.6

# Ensure the embeddings file exists
if not os.path.exists(embeddings_path):
    with open(embeddings_path, "wb") as f:
        pickle.dump([], f)

# Load embeddings
def load_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Send email notification
def send_email(to_email, student_name):
    subject = "Attendance Marked"
    body = f"Hi {student_name},\n\nYour attendance has been marked successfully.\n\nBest regards,\nAttendance System"
    msg = MIMEMultipart()
    msg['From'] = email_sender
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_sender, email_password)
            server.send_message(msg)
        print(f"Email sent to {student_name} ({to_email}).")
    except Exception as e:
        print(f"Failed to send email to {student_name}: {e}")

# Find closest match
def find_closest_match(face_embedding, embeddings):
    if not face_embedding:
        print("Error: Invalid face embedding.")
        return None

    face_embedding = face_embedding[0]["embedding"]  # Extract numerical embedding
    similarities = []
    for item in embeddings:
        db_embedding = item["embedding"]
        try:
            similarity = cosine_similarity([face_embedding], [db_embedding])[0][0]
            similarities.append((similarity, item))
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            continue

    similarities = sorted(similarities, key=lambda x: x[0], reverse=True)
    if similarities and similarities[0][0] >= SIMILARITY_THRESHOLD:
        return similarities[0][1]
    return None

# Log attendance
def log_attendance(student_id, class_name):
    # Load attendance
    if not os.path.exists(attendance_file) or os.stat(attendance_file).st_size == 0:
        attendance = pd.DataFrame(columns=["ID", "Attendance", "Class", "Time"])
    else:
        attendance = pd.read_csv(attendance_file)

    # Check and log attendance
    if student_id not in attendance["ID"].values:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {
            "ID": student_id,
            "Attendance": "Present",
            "Class": class_name,
            "Time": current_time
        }
        attendance = pd.concat([attendance, pd.DataFrame([new_entry])], ignore_index=True)
        attendance.to_csv(attendance_file, index=False)
        print(f"[ATTENDANCE LOGGED] {student_id} ({class_name}).")
    else:
        return False  # Return False to avoid multiple logs
    return True

import subprocess
from flask import Flask, render_template, request, session

app = Flask(__name__)

# Set a secret key for session handling
app.secret_key = os.urandom(24)

# Admin credentials (for simplicity, hardcoded here; you can use a database)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

@app.route('/admin-dashboard')
def admin_dashboard():
    return render_template('admin_dashboard.html')


@app.route('/admin-login', methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Validate admin credentials
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            return "Invalid admin credentials", 401

    return render_template("admin_login.html")

@app.route("/submit-student-details", methods=["POST"])
def submit_student_details():
    # Retrieve form data
    name = request.form["name"]
    student_id = request.form["student_id"]
    email = request.form["email"]
    section = request.form["section"]
    num_samples = request.form.get("num_samples", 30)  # Default to 30 if not provided

    # Run detection.py with the required arguments
    try:
        subprocess.run(
            [
                "python", "detection.py",
                "--name", name,
                "--student_id", student_id,
                "--email", email,
                "--section", section,
                "--num_samples", str(num_samples)
            ],
            check=True
        )
        # Redirect to a confirmation page
        return render_template("submit_student_details.html", name=name)
    except subprocess.CalledProcessError as e:
        return f"Error capturing images: {str(e)}", 500

import subprocess

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET"])
def student_details():
    return render_template("student_details.html")


@app.route('/detect', methods=["POST"])
def detect():
    image_data = request.form.get("image")
    if not image_data:
        return "No image data received.", 400

    try:
        image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        temp_image_path = "temp_frame.jpg"
        image.save(temp_image_path)

        face_embedding = DeepFace.represent(img_path=temp_image_path, model_name="Facenet", enforce_detection=False)
        embeddings = load_embeddings(embeddings_path)

        match = find_closest_match(face_embedding, embeddings)
        if match:
            log_attendance(match["student_id"], embeddings)
            return {"status": "success", "message": f"Match found: {match['name']}"}, 200
        else:
            return {"status": "failure", "message": "No matching face found."}, 404

    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route('/capture', methods=['POST'])
def capture_images():
    try:
        # Capture the section from the form input
        section = request.form.get('section', 'DefaultSection')
        
        # Pass the section as an argument to the detection script
        subprocess.run(['python', 'detection.py', section], check=True)
        
        return "Images captured successfully and saved in the dataset!", 200
    except subprocess.CalledProcessError as e:
        return f"Error running detection script: {str(e)}", 500
    
@app.route('/files')
def files():
    # Load both CSV files into dataframes
    data_4CSEA = pd.read_csv("4CSEA.csv")
    data_4CSEB = pd.read_csv("4CSEB.csv")

    # Convert dataframes to HTML tables
    table_4CSEA = data_4CSEA.to_html(classes="table table-striped table-bordered", index=False)
    table_4CSEB = data_4CSEB.to_html(classes="table table-striped table-bordered", index=False)

    # Pass the tables to the template
    return render_template("files.html", table_4CSEA=table_4CSEA, table_4CSEB=table_4CSEB)


if __name__ == "__main__":
    app.run(debug=True)
