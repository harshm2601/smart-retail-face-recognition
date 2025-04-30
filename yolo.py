import cv2
import face_recognition
import numpy as np
import sqlite3
from ultralytics import YOLO
import os
from pathlib import Path

# Initialize database connection
conn = sqlite3.connect('customer_faces_yolo.db')
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

# Initialize YOLOv8 model for face detection
model = YOLO('yolov8n-face.pt')

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

def process_video(input_path, output_path, model, known_customers):
    video_capture = cv2.VideoCapture(str(input_path))
    
    # Get video properties for output
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Use actual frame dimensions:
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    # OR uncomment this to ensure the written frames match (640, 360):
    # frame = cv2.resize(frame, (640, 360))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame for faster processing
        # frame = cv2.resize(frame, (640, 360))

        # Detect faces using YOLOv8
        # results = model(frame)
        # print(results)
        # Detect faces using YOLOv8
        results = model(frame)

        for result in results[0].boxes.data:
            x1, y1, x2, y2, confidence, cls = result
            if confidence < 0.5:  # Add confidence threshold
                continue

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            face_location = (y1, x2, y2, x1)

            # Extract the face
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            if face_image.size == 0:
                continue

            # Convert the face image to RGB format
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Compute face encoding
            face_encodings = face_recognition.face_encodings(face_image_rgb)
            if not face_encodings:
                continue
            face_encoding = face_encodings[0]

            # Recognize customer
            id, name = recognize_customer(face_encoding, known_customers)
            if id is None:
                # Register new customer
                name = f'Customer_{len(known_customers) + 1}'
                register_new_customer(name, face_encoding)
                known_customers = get_all_customers()
                id, name = recognize_customer(face_encoding, known_customers)

            # Annotate frame with customer ID
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {id}, Name: {name}', (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Write the frame to output file
        out.write(frame)

    video_capture.release()
    out.release()

# Create output directory if it doesn't exist
input_folder = Path('vdos')
output_folder = Path('output_yolov8')
output_folder.mkdir(exist_ok=True)

# Initialize model and load customers
model = YOLO('yolov8n-face.pt')
known_customers = get_all_customers()

# Process all videos in the input folder
for video_file in input_folder.glob('*.mp4'):
    output_path = output_folder / f'processed_{video_file.name}'
    process_video(video_file, output_path, model, known_customers)

cv2.destroyAllWindows()
conn.close()
