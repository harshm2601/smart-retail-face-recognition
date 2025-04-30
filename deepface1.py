import cv2
import numpy as np
import sqlite3
import os
from pathlib import Path
from deepface import DeepFace

# Initialize database connection
conn = sqlite3.connect('customer_faces_deep.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    embedding BLOB
)
''')
conn.commit()

def get_face_embedding(face_img):
    try:
        embedding_objs = DeepFace.represent(face_img, model_name="VGG-Face", enforce_detection=False)
        # Access the first returned embedding dictionary
        embedding = embedding_objs[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        return None

def register_new_customer(name, face_embedding):
    cursor.execute('INSERT INTO customers (name, embedding) VALUES (?, ?)', 
                  (name, face_embedding.tobytes()))
    conn.commit()

def get_all_customers():
    cursor.execute('SELECT id, name, embedding FROM customers')
    customers = []
    for row in cursor.fetchall():
        id, name, embedding = row
        embedding = np.frombuffer(embedding, dtype=np.float64)
        customers.append((id, name, embedding))
    return customers

def recognize_customer(face_embedding, known_customers, threshold=0.3):
    for id, name, db_embedding in known_customers:
        distance = np.linalg.norm(db_embedding - face_embedding)
        if distance < threshold:
            return id, name
    return None, None

def process_video(input_path, output_path, known_customers):
    video_capture = cv2.VideoCapture(str(input_path))
    
    # Get video properties
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces using DeepFace
        # try:
        faces = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
        # print("faces:" ,faces)
        for face in faces:
            # print("face:", face)
            # print("face['facial_area']:", face['facial_area'])
            # print("face['facial_area'].values():", face['facial_area'].values())
            x, y, w, h = list(face['facial_area'].values())[:4]
            face_img = frame[y:y+h, x:x+w]
            
            # Get face embedding
            face_embedding = get_face_embedding(face_img)
            if face_embedding is None:
                continue

            # Recognize customer
            id, name = recognize_customer(face_embedding, known_customers)
            if id is None:
                name = f'Customer_{len(known_customers) + 1}'
                register_new_customer(name, face_embedding)
                known_customers = get_all_customers()
                id, name = recognize_customer(face_embedding, known_customers)

            # Draw rectangle and name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {id}, Name: {name}', 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (36,255,12), 2)

        # except Exception as e:
        #     print(f"Error processing frame: {e}")
        #     continue

        out.write(frame)

    video_capture.release()
    out.release()

# Create output directory if it doesn't exist
input_folder = Path('vdos')
output_folder = Path('output_deep')
output_folder.mkdir(exist_ok=True)

# Load known customers
known_customers = get_all_customers()

# Process all videos in the input folder
for video_file in input_folder.glob('*.mp4'):
    output_path = output_folder / f'deepface_processed_{video_file.name}'
    process_video(video_file, output_path, known_customers)

cv2.destroyAllWindows()
conn.close()
