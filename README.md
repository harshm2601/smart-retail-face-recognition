# Face Recognition Customer Tracking System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python 3.7+"/>
  <img src="https://img.shields.io/badge/OpenCV-4.5+-green.svg" alt="OpenCV 4.5+"/>
  <img src="https://img.shields.io/badge/DeepFace-0.0.75+-orange.svg" alt="DeepFace"/>
  <img src="https://img.shields.io/badge/YOLOv8-latest-yellow.svg" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/face--recognition-1.3.0-purple.svg" alt="face-recognition"/>
</div>

<p align="center">A comprehensive system for tracking and identifying customers in video footage using state-of-the-art face recognition technologies.</p>

## 📋 Overview

This project implements multiple face recognition approaches to identify customers in video footage. It automatically detects faces, recognizes returning customers, and registers new ones in real-time. Each implementation offers different trade-offs in terms of speed, accuracy, and resource usage.

## ✨ Features

- **Multiple face detection & recognition models:**
  - Face Recognition library (dlib-based)
  - DeepFace with VGG-Face model
  - DeepFace with Facenet512 and folder storage
  - YOLOv8 face detection with face_recognition
- **Automated customer tracking and registration**
- **Persistent customer database using SQLite**
- **Real-time video processing with visual annotations**
- **Processing of multiple video files in batch**
- **Flexible configuration options for performance tuning**

## 🔍 Method Comparison

| Method | Detection | Recognition | Speed | Accuracy | Resource Usage | Database |
|--------|-----------|------------|-------|---------|----------------|---------|
| **face_recognition** | HOG | dlib (128-dim) | Medium | Good | Low | SQLite (embeddings) |
| **DeepFace VGG** | RetinaFace | VGG-Face | Slow | Very High | High | SQLite (embeddings) |
| **DeepFace Folder** | OpenCV | Facenet512 | Medium | High | Medium | Folder + SQLite |
| **YOLOv8** | YOLOv8 | dlib (128-dim) | Fast | Good | Medium | SQLite (embeddings) |

### Analysis

- **face_recognition** (main.py): Best for simple deployments with balanced performance
- **DeepFace VGG** (deepface1.py): Highest recognition accuracy but computationally intensive
- **DeepFace Folder** (deepface2.py): Good for examining and reviewing detected faces
- **YOLOv8** (yolo.py): Best detection speed and confidence with good overall performance

## 🛠️ Installation

1. **Clone or download this repository:**
   ```
   git clone <repository-url>
   cd Redcone
   ```

2. **Install dependencies:**
   ```
   pip install opencv-python numpy face_recognition deepface ultralytics pandas
   ```
   
   Note: For face_recognition, you may need to install dlib first:
   - Windows: Follow instructions at [face_recognition installation guide](https://github.com/ageitgey/face_recognition#installation)
   - Linux: `pip install dlib face_recognition`

3. **Download YOLOv8 model:**
   The YOLOv8 model will download automatically on first run, or you can pre-download it:
   ```
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt
   ```

## 📂 Project Structure

```
Redcone/
├── main.py                 # Implementation using face_recognition
├── deepface1.py            # Implementation using DeepFace with VGG-Face
├── deepface2.py            # Implementation using DeepFace with folder storage
├── yolo.py                 # Implementation using YOLOv8 and face_recognition
├── vdos/                   # Input video files
├── output/                 # Output for face_recognition method
├── output_deep/            # Output for DeepFace methods
├── output_yolov8/          # Output for YOLOv8 method
├── customer_folder/        # Folder-based storage for DeepFace2 approach
├── customer_faces.db       # SQLite database for face_recognition approach
├── customer_faces_deep.db  # SQLite database for DeepFace approach
└── customer_faces_yolo.db  # SQLite database for YOLOv8 approach
```

## 🚀 Usage

### Basic Usage:

1. **Place your video files:**
   Add your MP4 video files to the `vdos` directory.

2. **Choose an implementation and run:**

   ```bash
   # Using standard face_recognition:
   python main.py
   
   # Using DeepFace with VGG-Face:
   python deepface1.py
   
   # Using DeepFace with folder storage:
   python deepface2.py
   
   # Using YOLOv8 face detection:
   python yolo.py
   ```

3. **View results:**
   Processed videos will be saved in the corresponding output directory:
   - `output/` for face_recognition
   - `output_deep/` for DeepFace implementations
   - `output_yolov8/` for YOLOv8 implementation

### Advanced Configuration:

#### Adjust face recognition tolerance:
- In main.py and yolo.py:
  ```python
  # Lower values = stricter matching
  def recognize_customer(face_encoding, known_customers, tolerance=0.6):
  ```

- In deepface1.py:
  ```python
  # Lower values = stricter matching
  def recognize_customer(face_embedding, known_customers, threshold=0.3):
  ```

#### Change YOLOv8 detection confidence:
- In yolo.py:
  ```python
  # Higher values = fewer but more confident detections
  if confidence < 0.5:  # Adjust this threshold
  ```

#### Switch DeepFace detector backend:
- In deepface1.py:
  ```python
  # Options: 'opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib'
  faces = DeepFace.extract_faces(frame, detector_backend='retinaface', enforce_detection=False)
  ```

## ⚙️ How It Works

1. **Face Detection**: Identifies faces in each video frame
2. **Feature Extraction**: Creates numerical embeddings to represent each face
3. **Face Matching**: Compares embeddings against database of known customers
4. **Customer Registration**: Adds new faces to database when no match is found
5. **Video Annotation**: Draws bounding boxes and labels on detected faces
6. **Output Generation**: Creates annotated video files showing recognition results

## 🔒 Database Structure

- **face_recognition & YOLOv8**: Store 128-dimensional face embeddings in SQLite
- **DeepFace VGG**: Store VGG-Face embeddings (higher dimensional) in SQLite
- **DeepFace Folder**: Store actual face images on disk with folder organization

## 📜 License

This project is for educational and internal use only.

## 👨‍💻 Credits

Developed as part of an internship task at Redcone.
