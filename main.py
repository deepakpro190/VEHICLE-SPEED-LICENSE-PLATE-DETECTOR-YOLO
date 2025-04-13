import cv2
import torch
import numpy as np
import pytesseract
from ultralytics import YOLO
from collections import defaultdict, deque
from tqdm import tqdm
import csv

# ✅ Configuration
#VIDEO_INPUT = "v2.webm" 
#VIDEO_OUTPUT = "output/23-03-2025_01.mp4" 
#CSV_OUTPUT = "speeding_vehicles.csv" 
from huggingface_hub import hf_hub_download


# Download YOLO weights from Hugging Face
plate_model_path = hf_hub_download(repo_id="deepakpro190/my-yolo-plate-detect-model", filename="best.pt")
vehicle_model = YOLO("yolov8n.pt")  # Local YOLO model for vehicle detection
plate_model = YOLO(plate_model_path)  # HF Model for license plate detection


dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}
import string

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.
    Returns True if valid, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in "0123456789" or text[2] in dict_char_to_int.keys()) and \
       (text[3] in "0123456789" or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    return False
def format_license(text):
    """
    Corrects commonly misread characters in license plates.
    """
    formatted_text = ""
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 
               5: dict_int_to_char, 6: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int}

    for i in range(len(text)):
        if i in mapping and text[i] in mapping[i]:
            formatted_text += mapping[i][text[i]]
        else:
            formatted_text += text[i]

    return formatted_text




# ✅ Perspective Transformation Setup
SOURCE = np.array([[320, 150], [960, 150], [1250, 720], [50, 720]])  
TARGET_WIDTH, TARGET_HEIGHT = 80, 50
TARGET = np.array([
    [0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]
])

class ViewTransformer:
    def __init__(self, source, target):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points):
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

view_transformer = ViewTransformer(SOURCE, TARGET)

# ✅ Speed Tracking Storage
vehicle_speeds = defaultdict(lambda: 0)
prev_positions = {}
speed_history = defaultdict(lambda: deque(maxlen=10))

def calculate_speed(track_id, new_x, new_y, fps):
    global prev_positions, speed_history

    transformed_point = view_transformer.transform_points(np.array([[new_x, new_y]]))
    if transformed_point.size == 0:
        return 0  

    real_new_x, real_new_y = transformed_point[0]

    if track_id in prev_positions:
        old_x, old_y, _ = prev_positions[track_id]
        distance = np.sqrt((real_new_x - old_x) ** 2 + (real_new_y - old_y) ** 2)
        time_diff = 1 / fps
        speed = (distance / time_diff) * 3.6  
        # 🔹 Cap Unrealistic Speed Variations
        if speed > 120:
            speed = speed_history[track_id][-1] if speed_history[track_id] else 80
        # 🔹 Apply Rolling Average
        speed_history[track_id].append(speed)
        avg_speed = sum(speed_history[track_id]) / len(speed_history[track_id])
    else:
        avg_speed = 0

    prev_positions[track_id] = (real_new_x, real_new_y, fps)
    return avg_speed

# ✅ OCR License Plate Extraction
import pytesseract

def extract_license_plate_text(plate_img):
    """Extracts text from license plate image using OCR."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    plate_text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    plate_text = plate_text.strip()
    
    # Correct OCR misread characters
    corrected_plate = format_license(plate_text)
    
    # Validate license plate format
    if license_complies_format(corrected_plate):
        return corrected_plate
    return None  # Ignore invalid plates

import csv


def write_csv(speeding_vehicles, output_path):
    """Saves the detected vehicles & plates to a CSV file."""
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['car_id', 'speed_kmh', 'license_number'])  # CSV header

        for vehicle in speeding_vehicles:
            if len(vehicle) == 3:  # Ensure correct data format
                track_id, speed, plate_text = vehicle
                writer.writerow([track_id, speed, plate_text])



def run_detection(VIDEO_INPUT, VIDEO_OUTPUT, csv_output_path, vehicle_model, plate_model):
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {VIDEO_INPUT}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (frame_width, frame_height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    speeding_vehicles = []

    with tqdm(total=total_frames, desc="Processing Video", ncols=100) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  

            # 🔹 Step 1: Detect Vehicles using YOLOv8
            vehicle_results = vehicle_model(frame)
            vehicle_detections = vehicle_results[0]

            if vehicle_detections.boxes is not None:
                boxes = vehicle_detections.boxes.xyxy.cpu().numpy()
                ids = range(len(boxes))

                for track_id, (x1, y1, x2, y2) in zip(ids, boxes):
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    speed = calculate_speed(track_id, center_x, center_y, fps)
                    vehicle_speeds[track_id] = speed

                    # Determine Box Color (Green if speed ≤ 60 km/h, Red otherwise)
                    box_color = (0, 255, 0) if speed <= 60 else (0, 0, 255)

                    # 🟥 DRAW Bounding Box & Speed Text
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)
                    cv2.putText(frame, f"ID {track_id} | {int(speed)} km/h", 
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

                    # 🔹 Step 2: Detect License Plates
                    if speed > 60:
                        plate_results = plate_model(frame)
                        for plate in plate_results[0].boxes.xyxy.cpu().numpy():
                            px1, py1, px2, py2 = map(int, plate)
                            plate_crop = frame[py1:py2, px1:px2]
                            plate_text = extract_license_plate_text(plate_crop)

                            # 🟥 DRAW License Plate Box & Text
                            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                            cv2.putText(frame, plate_text, (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                            speeding_vehicles.append([track_id, int(speed), plate_text])

            out.write(frame)
            pbar.update(1)
    
    cap.release()
    out.release()
    write_csv(speeding_vehicles,csv_output_path)
    print(f"✅ Processing complete. Output saved to {VIDEO_OUTPUT}")
