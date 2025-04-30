import cv2
import numpy as np
import sqlite3
import os
from pathlib import Path
from deepface import DeepFace
import pandas as pd

# Initialize database connection
conn = sqlite3.connect('customer_faces_deep.db')
cursor = conn.cursor()

# Create table to store customer names and their image directory (path to subfolder)
cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    image_folder TEXT
)
''')
conn.commit()

# Register new customer by storing their image folder in the database
def register_new_customer(name, image_folder):
    cursor.execute('INSERT INTO customers (name, image_folder) VALUES (?, ?)', (name, image_folder))
    conn.commit()

# Retrieve all customer data from the database
def get_all_customers():
    cursor.execute('SELECT id, name, image_folder FROM customers')
    return cursor.fetchall()

def process_video(input_path, output_path, db_path):
    video_capture = cv2.VideoCapture(str(input_path))
    if not video_capture.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    customer_count = 1

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

        # Detect faces and match using DeepFace
        

        faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
        if not faces:
            out.write(frame)
            continue

        for face_data in faces:
            face_img = face_data['face']
            facial_area = face_data['facial_area']
            
            # Save detected face temporarily
            temp_face_path =  "temp_face.jpg"
            print("Face image shape:", face_img.shape)
            print("Face image pixel values:", np.mean(face_img))

            # cv2.imshow("Extracted Face", face_img)
            # cv2.waitKey(0)

            cv2.imwrite(temp_face_path, face_img)

            try:
                # Use the file path instead of numpy array
                results = DeepFace.find(img_path=temp_face_path, db_path=db_path, model_name='Facenet512', enforce_detection=False)
            except Exception as e:
                print(f"Error processing frame: {e}")
                results = []  # Set results to empty list in case of an error
            
            # Ensure results is a dataframe
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], pd.DataFrame):
                results = results[0]
            else:
                results = pd.DataFrame()

            if not results.empty:
                customer_info = results.iloc[0]
                name = customer_info['identity'].split(os.sep)[-2]  # Extract folder name
            else:
                # Generate a new customer ID
                name = f'Customer_{customer_count}'
                customer_count += 1

                # Create a unique folder for new customer
                customer_folder = os.path.join(db_path, name)
                os.makedirs(customer_folder, exist_ok=True)

                # Use a counter for unique image names
                img_counter = len(os.listdir(customer_folder)) + 1
                image_path = os.path.join(customer_folder, f"{name}_{img_counter}.jpg")

                # Save the new face image in the customer's folder
                cv2.imwrite(image_path, face_img)

                # Register the new customer in the database
                register_new_customer(name, customer_folder)

            # Annotate frame with customer name
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Clean up
        if os.path.exists(temp_face_path):
            os.remove(temp_face_path)

            

        out.write(frame)

    video_capture.release()
    out.release()

# Paths
input_folder = Path('vdos')
output_folder = Path('output_deep')
output_folder.mkdir(exist_ok=True)

customer_folder = Path('customer_folder')
customer_folder.mkdir(exist_ok=True)

# Process all videos
for video_file in input_folder.glob('*.mp4'):
    output_path = output_folder / f'processed_{video_file.name}'
    process_video(video_file, output_path, 'customer_folder')

cv2.destroyAllWindows()
conn.close()
