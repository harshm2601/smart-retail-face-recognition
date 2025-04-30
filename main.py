import cv2
import face_recognition
import numpy as np
import sqlite3
import os
from pathlib import Path

# Initialize database connection
conn = sqlite3.connect('customer_faces.db')
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

def register_new_customer(name, face_encoding):
    cursor.execute('INSERT INTO customers (name, embedding) VALUES (?, ?)', (name, face_encoding.tobytes()))
    conn.commit()

def get_all_customers():
    cursor.execute('SELECT id, name, embedding FROM customers')
    rows = cursor.fetchall()
    customers = []
    for row in rows:
        id, name, embedding = row
        embedding = np.frombuffer(embedding, dtype=np.float64)
        customers.append((id, name, embedding))
    return customers

def recognize_customer(face_encoding, known_customers, tolerance=0.6):
    for id, name, db_encoding in known_customers:
        distance = np.linalg.norm(db_encoding - face_encoding)
        if distance < tolerance:
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

        # Detect faces
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            id, name = recognize_customer(face_encoding, known_customers)
            if id is None:
                # Register new customer
                name = f'Customer_{len(known_customers) + 1}'
                register_new_customer(name, face_encoding)
                known_customers = get_all_customers()  # Update known customers
                id, name = recognize_customer(face_encoding, known_customers)

            # Annotate frame with customer ID
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {id}, Name: {name}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Write frame to output
        out.write(frame)

    video_capture.release()
    out.release()

# Create output directory if it doesn't exist
input_folder = Path('vdos')
output_folder = Path('output')
output_folder.mkdir(exist_ok=True)

# Load known customers
known_customers = get_all_customers()

# Process all videos in the input folder
for video_file in input_folder.glob('*.mp4'):
    output_path = output_folder / f'processed_{video_file.name}'
    process_video(video_file, output_path, known_customers)

cv2.destroyAllWindows()
conn.close()
