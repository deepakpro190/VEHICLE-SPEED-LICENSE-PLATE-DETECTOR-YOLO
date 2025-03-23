'''import cv2
import torch
import numpy as np
import keras_ocr
from ultralytics import YOLO
from tqdm import tqdm
from sort import Sort  # SORT Tracker

# ✅ Set GPU if available
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {torch_device}")

# ✅ Load YOLO models
vehicle_model = YOLO("yolov8l.pt")  # Vehicle detection
plate_model = YOLO("runs/detect/trained12/weights/best.pt")  # License plate detection
print("Models loaded successfully.")

# ✅ Initialize SORT tracker for vehicles & plates
vehicle_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
plate_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# ✅ Initialize Keras-OCR for license plate reading
ocr_pipeline = keras_ocr.pipeline.Pipeline()

# ✅ Define vehicle class IDs (excluding people)
ALLOWED_VEHICLE_CLASSES = {2, 3, 5, 7}  # Car, Motorcycle, Bus, Truck

def validate_image(image):
    """Ensure image has non-zero width & height before processing."""
    if image is None or image.size == 0:
        return False
    return True

def process_vehicle_frame(vehicle_crop):
    """Runs plate detection only on cropped vehicle images."""
    if not validate_image(vehicle_crop):
        return []  # Skip invalid images

    plate_results = plate_model(vehicle_crop)
    detected_plates = []
    
    for result in plate_results:
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = float(result.boxes.conf[i].item())
            if confidence > 0.3:
                detected_plates.append((x1, y1, x2, y2))
    
    return detected_plates

def run_license_plate_tracking(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc="Processing Video", ncols=100) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # 🔹 Step 1: Detect Vehicles using YOLO
            vehicle_results = vehicle_model(frame)
            vehicle_detections = []
            for result in vehicle_results:
                for i, box in enumerate(result.boxes.xyxy):
                    class_id = int(result.boxes.cls[i].item())
                    if class_id not in ALLOWED_VEHICLE_CLASSES:
                        continue  # Ignore people, bicycles, etc.

                    x1, y1, x2, y2 = map(int, box[:4])
                    confidence = float(result.boxes.conf[i].item())
                    if confidence > 0.4:
                        vehicle_detections.append([x1, y1, x2, y2, confidence])

            # 🔹 Step 2: Track Vehicles using SORT
            tracked_vehicles = vehicle_tracker.update(np.array(vehicle_detections)) if vehicle_detections else []

            # 🔹 Step 3: Process Each Vehicle Image for Plate Detection
            for track in tracked_vehicles:
                vx1, vy1, vx2, vy2, track_id = map(int, track)

                # Crop the vehicle image
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                if not validate_image(vehicle_crop):
                    continue  # Skip if invalid crop
                
                # Detect plates inside the vehicle image
                detected_plates = process_vehicle_frame(vehicle_crop)
                
                # Draw vehicle bounding box
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicle {track_id}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # 🔹 Step 4: Process Plates & Apply OCR
                for px1, py1, px2, py2 in detected_plates:
                    # Convert plate coords to original video frame scale
                    px1, px2 = px1 + vx1, px2 + vx1
                    py1, py2 = py1 + vy1, py2 + vy1

                    # Draw license plate bounding box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    cv2.putText(frame, "Plate", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    # Extract license plate number using Keras-OCR
                    plate_crop = frame[py1:py2, px1:px2]
                    if not validate_image(plate_crop):
                        continue  # Skip invalid plates
                    
                    plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)  # Convert for OCR
                    plate_images = [plate_crop]

                    try:
                        plate_text_predictions = ocr_pipeline.recognize(plate_images)
                        if plate_text_predictions and plate_text_predictions[0]:
                            plate_text = "".join([text[0] for text in plate_text_predictions[0]])
                            cv2.putText(frame, plate_text, (px1, py2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    except Exception as e:
                        print(f"Error in OCR: {str(e)}")

            out.write(frame)  # Save frame
            pbar.update(1)
    
    cap.release()
    out.release()
    print(f"Processing complete. Output saved to {output_video_path}")

# ✅ Run tracking
run_license_plate_tracking("v2.webm", "output/plate_tracking3.mp4")
'''
import cv2
import torch
import numpy as np
import keras_ocr
import argparse
from ultralytics import YOLO
from tqdm import tqdm
from sort import Sort  # SORT Tracker
from concurrent.futures import ThreadPoolExecutor
import os

# ✅ Allow TensorFlow (Keras-OCR) to run on CPU to avoid GPU contention
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Set GPU if available for YOLO models
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for YOLO: {torch_device}")

# ✅ Load YOLO models
vehicle_model = YOLO("yolov8n.pt")  # Vehicle detection
plate_model = YOLO("runs/detect/trained12/weights/best.pt")  # License plate detection
print("Models loaded successfully.")

# ✅ Initialize SORT tracker for vehicles & plates
vehicle_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
plate_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# ✅ Initialize Keras-OCR for license plate reading
ocr_pipeline = keras_ocr.pipeline.Pipeline()

# ✅ Define vehicle class IDs (excluding people)
ALLOWED_VEHICLE_CLASSES = {2, 3, 5, 7}  # Car, Motorcycle, Bus, Truck

def validate_image(image):
    """Ensure image has non-zero width & height before processing."""
    return image is not None and image.size > 0

def process_vehicle_frame(vehicle_crop):
    """Runs plate detection only on cropped vehicle images."""
    if not validate_image(vehicle_crop):
        return []

    plate_results = plate_model(vehicle_crop, device=torch_device)
    detected_plates = []
    
    for result in plate_results:
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = float(result.boxes.conf[i].item())
            if confidence > 0.3:
                detected_plates.append((x1, y1, x2, y2))
    
    return detected_plates

def run_license_plate_tracking(video_path, output_video_path, progress_callback=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with ThreadPoolExecutor(max_workers=4) as executor, tqdm(total=total_frames, desc="Processing Video", ncols=100) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # 🔹 Step 1: Detect Vehicles using YOLO
            vehicle_results = vehicle_model(frame, device=torch_device)
            vehicle_detections = []
            for result in vehicle_results:
                for i, box in enumerate(result.boxes.xyxy):
                    class_id = int(result.boxes.cls[i].item())
                    if class_id not in ALLOWED_VEHICLE_CLASSES:
                        continue  # Ignore people, bicycles, etc.

                    x1, y1, x2, y2 = map(int, box[:4])
                    confidence = float(result.boxes.conf[i].item())
                    if confidence > 0.4:
                        vehicle_detections.append([x1, y1, x2, y2, confidence])

            # 🔹 Step 2: Track Vehicles using SORT
            tracked_vehicles = vehicle_tracker.update(np.array(vehicle_detections)) if vehicle_detections else []
            
            futures = {}
            for track in tracked_vehicles:
                vx1, vy1, vx2, vy2, track_id = map(int, track)
                vehicle_crop = frame[vy1:vy2, vx1:vx2].copy()  # ✅ Prevent memory issues

                if not validate_image(vehicle_crop):
                    continue
                
                futures[executor.submit(process_vehicle_frame, vehicle_crop)] = (vx1, vy1, vx2, vy2, track_id)
            
            plate_crops = []
            plate_positions = []

            for future in futures:
                detected_plates = future.result()
                vx1, vy1, vx2, vy2, track_id = futures[future]
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicle {track_id}", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                for px1, py1, px2, py2 in detected_plates:
                    px1, px2 = px1 + vx1, px2 + vx1
                    py1, py2 = py1 + vy1, py2 + vy1
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    cv2.putText(frame, "Plate", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    plate_crop = frame[py1:py2, px1:px2]
                    if validate_image(plate_crop):
                        plate_crops.append(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
                        plate_positions.append((px1, py2))

            # ✅ Batch OCR processing
            if plate_crops:
                try:
                    batch_plate_texts = ocr_pipeline.recognize(plate_crops)
                    for text_predictions, (px1, py2) in zip(batch_plate_texts, plate_positions):
                        if text_predictions:
                            plate_text = "".join([text[0] for text in text_predictions])
                            cv2.putText(frame, plate_text, (px1, py2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                except Exception as e:
                    print(f"Error in batch OCR: {str(e)}")

            out.write(frame)
            pbar.update(1)
    
    cap.release()
    out.release()
    print(f"Processing complete. Output saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run license plate tracking on a video.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save output video.")

    args = parser.parse_args()
    run_license_plate_tracking(args.input, args.output)

'''
import cv2
import torch
import numpy as np
import keras_ocr
from ultralytics import YOLO
from tqdm import tqdm
from sort import Sort  # SORT Tracker
from collections import defaultdict, Counter
import os

# ✅ Ensure output directory exists
os.makedirs("output", exist_ok=True)

# ✅ Set GPU if available
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {torch_device}")

# ✅ Load YOLO models
vehicle_model = YOLO("yolov8l.pt")  # Vehicle detection
plate_model = YOLO("runs/detect/trained12/weights/best.pt")  # License plate detection
print("Models loaded successfully.")

# ✅ Initialize SORT tracker for vehicles & plates
vehicle_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
plate_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# ✅ Initialize Keras-OCR for license plate reading
ocr_pipeline = keras_ocr.pipeline.Pipeline()

# ✅ Define vehicle class IDs (excluding people)
ALLOWED_VEHICLE_CLASSES = {2, 3, 5, 7}  # Car, Motorcycle, Bus, Truck

# ✅ Speed calculation parameters
REAL_WORLD_DISTANCE_METERS = 10  # Assumed real-world distance between frames
FPS = 30  # Default FPS (will be updated from video)

# ✅ Storage for tracking speed & license plates
vehicle_speeds = {}  # track_id -> last position, last frame
plate_memory = defaultdict(list)  # track_id -> list of detected plates


def estimate_speed(track_id, vx1, vy1, vx2, vy2, frame_count, fps):
    """
    Estimates the speed of a tracked vehicle.
    """
    if track_id in vehicle_speeds:
        last_position, last_frame = vehicle_speeds[track_id]
        frame_diff = frame_count - last_frame

        if frame_diff > 0:
            pixel_distance = np.sqrt((vx1 - last_position[0]) ** 2 + (vy1 - last_position[1]) ** 2)
            meters_per_pixel = REAL_WORLD_DISTANCE_METERS / pixel_distance if pixel_distance else 0
            speed_mps = meters_per_pixel * (fps / frame_diff)  # Speed in meters per second
            speed_kmh = speed_mps * 3.6  # Convert to km/h
        else:
            speed_kmh = 0  # No movement detected

        vehicle_speeds[track_id] = ((vx1, vy1), frame_count)
        return speed_kmh
    else:
        vehicle_speeds[track_id] = ((vx1, vy1), frame_count)
        return 0  # Initial frame, speed unknown


def process_vehicle_frame(vehicle_crop):
    """
    Runs plate detection only on the cropped vehicle image.
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return []  # Skip processing empty vehicle images

    plate_results = plate_model(vehicle_crop)  # Detect plates only inside the vehicle image
    detected_plates = []

    for result in plate_results:
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = float(result.boxes.conf[i].item())
            if confidence > 0.3:  # Ensure high confidence plates
                detected_plates.append((x1, y1, x2, y2))

    return detected_plates


def get_consistent_plate(track_id):
    """
    Returns the most frequent plate number detected for a vehicle.
    """
    if track_id in plate_memory and plate_memory[track_id]:
        plate_counts = Counter(plate_memory[track_id])
        return plate_counts.most_common(1)[0][0]  # Most common plate detected
    return None


def run_license_plate_tracking(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    global FPS
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, FPS, (frame_width, frame_height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    with tqdm(total=total_frames, desc="Processing Video", ncols=100) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            frame_count += 1

            # 🔹 Step 1: Detect Vehicles using YOLO
            vehicle_results = vehicle_model(frame)
            vehicle_detections = []
            for result in vehicle_results:
                for i, box in enumerate(result.boxes.xyxy):
                    class_id = int(result.boxes.cls[i].item())
                    if class_id not in ALLOWED_VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box[:4])
                    confidence = float(result.boxes.conf[i].item())
                    if confidence > 0.4:
                        vehicle_detections.append([x1, y1, x2, y2, confidence])

            # 🔹 Step 2: Track Vehicles using SORT
            tracked_vehicles = vehicle_tracker.update(np.array(vehicle_detections)) if vehicle_detections else []

            # 🔹 Step 3: Process Each Vehicle Image for Plate Detection
            for track in tracked_vehicles:
                vx1, vy1, vx2, vy2, track_id = map(int, track)

                # Crop the vehicle image
                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                if vehicle_crop is None or vehicle_crop.size == 0:
                    continue  

                # Estimate speed
                speed_kmh = estimate_speed(track_id, vx1, vy1, vx2, vy2, frame_count, FPS)

                # Detect plates inside the vehicle image
                detected_plates = process_vehicle_frame(vehicle_crop)

                # Draw vehicle bounding box
                color = (0, 255, 0) if speed_kmh <= 80 else (0, 0, 255)  # Green if normal, Red if overspeed
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), color, 2)
                cv2.putText(frame, f"Vehicle {track_id} | {int(speed_kmh)} km/h", (vx1, vy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # 🔹 Step 4: Process Plates & Apply OCR
                for px1, py1, px2, py2 in detected_plates:
                    px1, px2 = px1 + vx1, px2 + vx1
                    py1, py2 = py1 + vy1, py2 + vy1

                    plate_crop = frame[py1:py2, px1:px2]
                    if plate_crop is None or plate_crop.size == 0:
                        continue  # Skip empty plate images

                    plate_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    plate_text_predictions = ocr_pipeline.recognize([plate_crop])

                    if plate_text_predictions and plate_text_predictions[0]:
                        plate_text = "".join([text[0] for text in plate_text_predictions[0]])
                        plate_memory[track_id].append(plate_text)

                        consistent_plate = get_consistent_plate(track_id)
                        if consistent_plate:
                            cv2.putText(frame, consistent_plate, (px1, py2 + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Processing complete. Output saved to {output_video_path}")


# ✅ Run tracking
run_license_plate_tracking("v2.webm", "output/plate_tracking2.1.mp4")
'''
