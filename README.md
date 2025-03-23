Here’s an **updated README.md** file incorporating **speed detection, perspective transformation, and license plate compliance features**:

---

# **🚗 Vehicle Speed & License Plate Detector using YOLO**  

This project is a **Streamlit-based web app** that detects vehicles, extracts license plates, estimates speed, and ensures license plate compliance using **YOLOv8**, **OpenCV**, and **Tesseract OCR**. The app processes an uploaded video and outputs a new video with detections, speed annotations, and an exported **CSV file** containing detected license plates and speeds.

---

## **🔹 Features**
✅ **Vehicle Detection** – Detects vehicles using **YOLOv8**  
✅ **License Plate Recognition** – Extracts text using **OCR & error correction**  
✅ **Speed Estimation** – Uses **perspective transformation & tracking**  
✅ **License Plate Format Compliance** – Ensures correct formatting  
✅ **Processed Video Output** – Overlays detections & speed on video  
✅ **CSV Export** – Saves results with vehicle speeds and license plates  

---

## **🚀 Demo**
Try the app on **Streamlit Cloud**:  
🔗 [Live Demo](https://your-app-url.streamlit.app/) *(Replace with actual link)*  

---

## **🛠 Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/deepakpro190/vehicle-speed-license-plate-detector-yolo.git
cd vehicle-speed-license-plate-detector-yolo
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

⚠ **Make sure you have Tesseract installed** (for OCR):  
```bash
sudo apt-get install tesseract-ocr  # Ubuntu/Linux
brew install tesseract              # macOS
```
For Windows, download from [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

### **3️⃣ Run the App**
```bash
streamlit run app.py
```

---

## **📂 Project Structure**
```
📦 vehicle-speed-license-plate-detector-yolo
 ┣ 📜 app.py                  # Streamlit UI
 ┣ 📜 main.py                 # YOLO detection & speed estimation
 ┣ 📜 requirements.txt         # Python dependencies
 ┣ 📂 models                   # YOLOv8 model weights
 ┣ 📂 data                     # Sample test videos
 ┣ 📂 outputs                  # Processed videos & CSVs
```

---

## **📌 How It Works**
### **1️⃣ Perspective Transformation (Bird’s Eye View)**
To **accurately estimate speed**, we apply a **perspective transformation** to map the video’s perspective to a **real-world coordinate system**.

🔹 **Source Points** (Original Video)  
🔹 **Target Points** (Mapped View)  

```python
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
```

---

### **2️⃣ Speed Estimation (Object Tracking)**
We track vehicles across frames and use a **rolling average** to stabilize speed estimations.

```python
def calculate_speed(track_id, new_x, new_y, fps):
    transformed_point = view_transformer.transform_points(np.array([[new_x, new_y]]))
    if transformed_point.size == 0:
        return 0  

    real_new_x, real_new_y = transformed_point[0]

    if track_id in prev_positions:
        old_x, old_y, _ = prev_positions[track_id]
        distance = np.sqrt((real_new_x - old_x) ** 2 + (real_new_y - old_y) ** 2)
        time_diff = 1 / fps
        speed = (distance / time_diff) * 3.6  # Convert m/s to km/h

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
```

---

### **3️⃣ License Plate Detection & Compliance**
OCR can **misread characters**, so we apply **correction mappings** and validate against a predefined license plate format.

#### **🔹 Character Correction for OCR Errors**
```python
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def format_license(text):
    formatted_text = ""
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 
               5: dict_int_to_char, 6: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int}

    for i in range(len(text)):
        if i in mapping and text[i] in mapping[i]:
            formatted_text += mapping[i][text[i]]
        else:
            formatted_text += text[i]

    return formatted_text
```

#### **🔹 License Plate Format Validation**
```python
import string

def license_complies_format(text):
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
```

---

## **📜 Example Output**
### **📜 CSV Format**
| Vehicle | Speed  | License Plate |
|---------|--------|--------------|
| Car     | 65 km/h | ABC123       |
| Truck   | 72 km/h | XYZ789       |

---

## **🔧 Troubleshooting**
### **1️⃣ OpenCV (`cv2`) Error?**
Install missing dependencies:
```bash
sudo apt-get install libgl1-mesa-glx -y
```

### **2️⃣ Tesseract OCR Not Found?**
Update the path in `app.py`:
```python
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
```

### **3️⃣ Video Not Opening in Streamlit?**
Ensure the file is closed before streaming:
```python
out.release()
cv2.destroyAllWindows()
```

