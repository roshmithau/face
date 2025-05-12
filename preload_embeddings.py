from deepface import DeepFace
import pickle
import os
import pandas as pd

# Paths
dataset_path = "dataset"
embeddings_path = "embeddings.pkl"
student_details_files = ["4CSEA.csv", "4CSEB.csv"]  # CSV files containing student details


# Load student details
def load_student_details(files):
    dataframes = []
    for file in files:
        if not os.path.exists(file):
            print(f"[ERROR] Student details file '{file}' not found.")
            return None
        dataframes.append(pd.read_csv(file))
    return pd.concat(dataframes, ignore_index=True)


# Generate embeddings
def preload_embeddings(dataset_path, model_name="Facenet"):
    student_details = load_student_details(student_details_files)
    if student_details is None:
        return
    
    embeddings = []
    print("[INFO] Generating embeddings...")

    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)

        if not os.path.isdir(class_path):
            continue

        for student_name in os.listdir(class_path):
            student_path = os.path.join(class_path, student_name)

            if not os.path.isdir(student_path):
                continue

            # Get correct student ID from CSV
            student_row = student_details[student_details["Name"] == student_name]
            if student_row.empty:
                print(f"[WARNING] No student ID for '{student_name}'. Skipping.")
                continue
            
            student_id = str(student_row.iloc[0]["ID"])  # Get student ID from CSV

            # Process images
            for file in os.listdir(student_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(student_path, file)
                    try:
                        print(f"[INFO] Processing image: {img_path}")
                        embedding_result = DeepFace.represent(
                            img_path=img_path,
                            model_name="Facenet",
                            enforce_detection=False
                        )

                        if isinstance(embedding_result, dict):
                            embedding_result = [embedding_result]  # Ensure it's a list

                        if isinstance(embedding_result, list) and len(embedding_result) > 0:
                            embedding = embedding_result[0].get("embedding", [])
                        else:
                            embedding = []

                        if not embedding:
                            print(f"[WARNING] No valid embedding for {img_path}. Skipping.")
                            continue

                        embeddings.append({
                            "student_id": student_id,  # Store actual student ID
                            "name": student_name,
                            "class_name": class_folder,
                            "embedding": embedding,
                            "img_path": img_path
                        })
                        print(f"[PROCESSED] {img_path} -> {student_id} ({student_name})")
                    
                    except Exception as e:
                        print(f"[ERROR] Failed to process {img_path}: {e}")

    # Check if any embeddings were generated
    if embeddings:
        print(f"[INFO] {len(embeddings)} embeddings generated.")
        with open(embeddings_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"[INFO] Embeddings saved to {embeddings_path}.")
    else:
        print("[INFO] No embeddings were generated.")


# Run the function
if __name__ == "__main__":
    preload_embeddings(dataset_path)
