'''import os
import cv2
import string
import easyocr
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))
        for frame_nmr in results:
            for car_id in results[frame_nmr]:
                bbox_score = results[frame_nmr][car_id]['license_plate'].get('bbox_score', None)
                f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                        car_id,
                                                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['car']['bbox']),
                                                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['license_plate']['bbox']),
                                                        bbox_score,
                                                        results[frame_nmr][car_id]['license_plate']['text'],
                                                        results[frame_nmr][car_id]['license_plate']['text_score'])
                        )

def read_license_plate(license_plate_crop_thresh, reader):
    results = reader.readtext(license_plate_crop_thresh)
    return (results[0][1], results[0][2]) if results else (None, None)

class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])

def run_inference_on_video(video_path, vehicle_model_path, plate_model_path, output_folder="output", max_frames=360):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)

    vehicle_model = YOLO(vehicle_model_path)
    plate_model = YOLO(plate_model_path)
    reader = easyocr.Reader(['en'], gpu=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(output_folder, 'output_video.mp4'), fourcc, 20.0, (frame_width, frame_height))

    results = {}
    frame_count = 0
    kalman = KalmanFilter()
    plate_memory = {}

    with tqdm(total=max_frames, desc="Processing video", ncols=100) as pbar:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval != 0:
                frame_count += 1
                continue
            
            vehicle_results = vehicle_model(frame)
            for vehicle_box in vehicle_results[0].boxes.xyxy:
                vx1, vy1, vx2, vy2 = map(int, vehicle_box)
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                plate_results = plate_model(vehicle_crop)
                
                for plate_box in plate_results[0].boxes.xyxy:
                    px1, py1, px2, py2 = map(int, plate_box)
                    
                    # Ensure bounding box coordinates are within vehicle_crop bounds
                    h, w, _ = vehicle_crop.shape
                    px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)
                    
                    if px1 >= px2 or py1 >= py2:  # Skip invalid bounding boxes
                        continue
                    
                    license_plate_crop = vehicle_crop[py1:py2, px1:px2]
                
                    # Ensure the crop is not empty
                    if license_plate_crop.size == 0:
                        continue
                
                    # Convert to grayscale safely
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
 
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                    
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh, reader)
                    
                    if license_plate_text:
                        car_id = f"car_{frame_count}_{vx1}_{vy1}"
                        plate_memory[car_id] = license_plate_text
                        results.setdefault(frame_count, {})[car_id] = {
                            'car': {'bbox': [vx1, vy1, vx2, vy2]},
                            'license_plate': {
                                'bbox': [px1, py1, px2, py2],
                                'bbox_score': license_plate_text_score,
                                'text': license_plate_text,
                                'text_score': license_plate_text_score
                            }
                        }
                        
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                        cv2.putText(frame, license_plate_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()
    write_csv(results, os.path.join(output_folder, 'detection_results.csv'))

# Run with example parameters
video_path = 'vid1.mp4'
vehicle_model_path = 'yolov8l.pt'
plate_model_path = 'runs/detect/trained12/weights/best.pt'

run_inference_on_video(video_path, vehicle_model_path, plate_model_path, output_folder="output", max_frames=360)
'''
'''

import os
import cv2
import easyocr
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from sort import Sort  # Object Tracker

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    print(f"Using Nvidia GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA-compatible GPU found. Using CPU.")

# Initialize YOLO models
vehicle_model = YOLO("yolov8l.pt")
plate_model = YOLO("runs/detect/model_15/weights/best.pt")

# Initialize Object Tracker
tracker = Sort()  

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=True)

# Character mappings for OCR correction
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

# Store tracked vehicle license plates
tracked_plates = {}

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))
        for frame_nmr in results:
            for car_id in results[frame_nmr]:
                bbox_score = results[frame_nmr][car_id]['license_plate'].get('bbox_score', None)
                f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                        car_id,
                                                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['car']['bbox']),
                                                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['license_plate']['bbox']),
                                                        bbox_score,
                                                        results[frame_nmr][car_id]['license_plate']['text'],
                                                        results[frame_nmr][car_id]['license_plate']['text_score'])
                        )

def read_license_plate(license_plate_crop_thresh):
    results = reader.readtext(license_plate_crop_thresh)
    return (results[0][1], results[0][2]) if results else (None, None)

def run_inference_on_video(video_path, output_folder="output", max_frames=360):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(output_folder, 'output_video.mp4'), fourcc, 20.0, (frame_width, frame_height))

    results = {}
    frame_count = 0

    with tqdm(total=max_frames, desc="Processing video", ncols=100) as pbar:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval != 0:  # Skip frames for efficiency
                continue
            
            detections = []

            # Detect vehicles
            vehicle_results = vehicle_model(frame)
            for vehicle_box in vehicle_results[0].boxes.xyxy:
                vx1, vy1, vx2, vy2 = map(int, vehicle_box)
                detections.append([vx1, vy1, vx2, vy2, 1])  # Format: [x1, y1, x2, y2, confidence]

            # Track vehicles
            trackers = tracker.update(np.array(detections))

            for track in trackers:
                vx1, vy1, vx2, vy2, track_id = map(int, track)
                vehicle_crop = frame[vy1:vy2, vx1:vx2]

                # Detect license plates inside the tracked vehicle
                plate_results = plate_model(vehicle_crop)
                for plate_box in plate_results[0].boxes.xyxy:
                    px1, py1, px2, py2 = map(int, plate_box)
                    license_plate_crop = vehicle_crop[py1:py2, px1:px2]
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text:
                        # Ensure consistent plate detection for the same vehicle
                        if track_id in tracked_plates:
                            prev_plate = tracked_plates[track_id]
                            if license_plate_text_score < prev_plate["score"]:  # Keep highest score
                                license_plate_text = prev_plate["text"]
                        tracked_plates[track_id] = {"text": license_plate_text, "score": license_plate_text_score}

                        # Save results
                        if frame_count not in results:
                            results[frame_count] = {}

                        results[frame_count][track_id] = {
                            'car': {'bbox': [vx1, vy1, vx2, vy2]},
                            'license_plate': {
                                'bbox': [px1, py1, px2, py2],
                                'bbox_score': license_plate_text_score,
                                'text': license_plate_text,
                                'text_score': license_plate_text_score
                            }
                        }

                        # Draw bounding boxes
                        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                        cv2.rectangle(frame, (px1+vx1, py1+vy1), (px2+vx1, py2+vy1), (0, 255, 0), 2)
                        cv2.putText(frame, f"Car {track_id}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(frame, license_plate_text, (px1+vx1, py1+vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()
    write_csv(results, os.path.join(output_folder, 'detection_results.csv'))


# Run inference
video_path = "vid1.mp4"
run_inference_on_video(video_path, output_folder="output", max_frames=360)
'''
'''
import os
import cv2
import pytesseract
import torch
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

# Set Tesseract Path (Update this path if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    print(f"Using Nvidia GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA-compatible GPU found. Using CPU.")

# Create output directory
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to enhance and read license plate text
def read_license_plate(license_plate_crop):
    """Enhances and extracts text from a license plate image."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

        # Sharpening filter
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)

        # Resize for better OCR accuracy
        resized = cv2.resize(sharpened, (300, 100))

        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # OCR with Tesseract (config optimized for license plates)
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        plate_text = pytesseract.image_to_string(processed, config=custom_config).strip()

        return plate_text if plate_text else "Unknown"
    
    except Exception as e:
        print(f"OCR Error: {e}")
        return "Unknown"

# Function to run detection
def run_inference(video_path, vehicle_model_path, plate_model_path, max_frames=360):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)  # Skip frames evenly

    # Load YOLO models
    vehicle_model = YOLO(vehicle_model_path)
    plate_model = YOLO(plate_model_path)

    frame_count = 0

    with tqdm(total=max_frames, desc="Processing video", ncols=100) as pbar:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval != 0:  # Skip frames
                frame_count += 1
                continue
            
            # Detect vehicles
            vehicle_results = vehicle_model(frame)
            for idx, vehicle_box in enumerate(vehicle_results[0].boxes.xyxy):
                vx1, vy1, vx2, vy2 = map(int, vehicle_box)
                vehicle_crop = frame[vy1:vy2, vx1:vx2]

                # Detect license plate within the cropped vehicle
                plate_results = plate_model(vehicle_crop)
                for plate_box in plate_results[0].boxes.xyxy:
                    px1, py1, px2, py2 = map(int, plate_box)
                    license_plate_crop = vehicle_crop[py1:py2, px1:px2]

                    # Run OCR on license plate
                    plate_text = read_license_plate(license_plate_crop)
                    plate_text = plate_text if plate_text else "Unknown"

                    # Draw bounding box around license plate
                    cv2.rectangle(vehicle_crop, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(vehicle_crop, plate_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Save the image with annotations
                    output_path = os.path.join(output_folder, f"frame_{frame_count}_vehicle_{idx}.jpg")
                    cv2.imwrite(output_path, vehicle_crop)

            frame_count += 1
            pbar.update(1)

    cap.release()

# Run with example parameters
video_path = 'vid1.mp4'
vehicle_model_path = 'yolov8l.pt'
plate_model_path = 'runs/detect/trained12/weights/best.pt'

run_inference(video_path, vehicle_model_path, plate_model_path, max_frames=360)
'''
'''
import os
import cv2
import torch
import keras_ocr
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    print(f"Using Nvidia GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA-compatible GPU found. Using CPU.")

# Create output directory
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Load Keras OCR pipeline
ocr_pipeline = keras_ocr.pipeline.Pipeline()

def read_license_plate(license_plate_crop):
    """Reads text from a license plate image using Keras OCR."""
    try:
        license_plate_crop = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)
        license_plate_crop = cv2.resize(license_plate_crop, (300, 100))
        predictions = ocr_pipeline.recognize([license_plate_crop])[0]
        plate_text = " ".join([text[0] for text in predictions]).strip()
        return plate_text if plate_text else "Unknown"
    except Exception as e:
        print(f"OCR Error: {e}")
        return "Unknown"

# Function to run detection and save video
def run_inference(video_path, vehicle_model_path, plate_model_path, output_video_path, max_frames=360):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)  # Skip frames evenly

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Load YOLO models
    vehicle_model = YOLO(vehicle_model_path)
    plate_model = YOLO(plate_model_path)

    frame_count = 0
    with tqdm(total=max_frames, desc="Processing video", ncols=100) as pbar:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval != 0:  # Skip frames
                frame_count += 1
                continue
            
            # Detect vehicles
            vehicle_results = vehicle_model(frame)
            for vehicle_box in vehicle_results[0].boxes.xyxy:
                vx1, vy1, vx2, vy2 = map(int, vehicle_box)
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                
                # Detect license plate within the cropped vehicle
                plate_results = plate_model(vehicle_crop)
                for plate_box in plate_results[0].boxes.xyxy:
                    px1, py1, px2, py2 = map(int, plate_box)
                    license_plate_crop = vehicle_crop[py1:py2, px1:px2]
                    
                    # Run OCR on license plate
                    plate_text = read_license_plate(license_plate_crop)
                    
                    # Remove special characters for file naming
                    sanitized_text = "".join(c for c in plate_text if c.isalnum()) or "unknown"
                    output_path = os.path.join(output_folder, f"plate_{sanitized_text}.jpg")
                    cv2.imwrite(output_path, license_plate_crop)
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                    cv2.rectangle(frame, (px1+vx1, py1+vy1), (px2+vx1, py2+vy1), (0, 255, 0), 2)
                    cv2.putText(frame, plate_text, (px1+vx1, py1+vy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            out.write(frame)  # Save frame to video
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()

# Run with example parameters
video_path = 'vid1.mp4'
vehicle_model_path = 'yolov8l.pt'
plate_model_path = 'runs/detect/trained12/weights/best.pt'
output_video_path = 'output/output_video.mp4'

run_inference(video_path, vehicle_model_path, plate_model_path, output_video_path, max_frames=360)
'''
import os
import cv2
import torch
import keras_ocr
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    print(f"Using Nvidia GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA-compatible GPU found. Using CPU.")

# Load Keras OCR pipeline
ocr_pipeline = keras_ocr.pipeline.Pipeline()

def read_license_plate(license_plate_crop):
    """Reads text from a license plate image using Keras OCR."""
    try:
        license_plate_crop = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)
        license_plate_crop = cv2.resize(license_plate_crop, (300, 100))
        predictions = ocr_pipeline.recognize([license_plate_crop])[0]
        plate_text = " ".join([text[0] for text in predictions]).strip()
        return plate_text if plate_text else "Unknown"
    except Exception as e:
        print(f"OCR Error: {e}")
        return "Unknown"

# Function to run detection and save video
def run_inference(video_path, vehicle_model_path, plate_model_path, output_video_path, max_frames=360):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)  # Skip frames evenly

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Load YOLO models
    vehicle_model = YOLO(vehicle_model_path)
    plate_model = YOLO(plate_model_path)

    plate_buffer = {}  # Store last seen plate to avoid flickering
    frame_count = 0
    with tqdm(total=max_frames, desc="Processing video", ncols=100) as pbar:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval != 0:  # Skip frames
                frame_count += 1
                continue
            
            # Detect vehicles
            vehicle_results = vehicle_model(frame)
            for vehicle_box in vehicle_results[0].boxes.xyxy:
                vx1, vy1, vx2, vy2 = map(int, vehicle_box)
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                
                # Detect license plate within the cropped vehicle
                plate_results = plate_model(vehicle_crop)
                for plate_box in plate_results[0].boxes.xyxy:
                    px1, py1, px2, py2 = map(int, plate_box)
                    license_plate_crop = vehicle_crop[py1:py2, px1:px2]
                    
                    # Run OCR on license plate
                    plate_text = read_license_plate(license_plate_crop)
                    
                    # Smooth bounding box
                    plate_id = (vx1, vy1, vx2, vy2)  # Use vehicle bbox as key
                    if plate_id in plate_buffer:
                        prev_text = plate_buffer[plate_id]["text"]
                        plate_buffer[plate_id]["count"] += 1
                        if plate_buffer[plate_id]["count"] > 3:  # Stabilize text
                            plate_buffer[plate_id]["text"] = plate_text
                    else:
                        plate_buffer[plate_id] = {"text": plate_text, "count": 1}
                    
                    stable_text = plate_buffer[plate_id]["text"]
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                    cv2.rectangle(frame, (px1+vx1, py1+vy1), (px2+vx1, py2+vy1), (0, 255, 0), 2)
                    cv2.putText(frame, stable_text, (px1+vx1, py1+vy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            out.write(frame)  # Save frame to video
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    out.release()

# Run with example parameters
video_path = 'vid1.mp4'
vehicle_model_path = 'yolov8l.pt'
plate_model_path = 'runs/detect/trained12/weights/best.pt'
output_video_path = 'output/output_video.mp4'

run_inference(video_path, vehicle_model_path, plate_model_path, output_video_path, max_frames=360)
