'''import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from main import run_detection  # Import your main detection function

# ✅ Streamlit UI Setup
st.set_page_config(page_title="Vehicle Speed & License Plate Detection", layout="wide")
st.title("🚗 Speeding Vehicle Detection with License Plate Recognition")

# ✅ Hugging Face Model Download
@st.cache_resource
def load_yolo_models():
    plate_model_path = hf_hub_download(repo_id="deepakpro190/my-yolo-plate-detect-model", filename="best.pt")
    return YOLO("yolov8n.pt"), YOLO(plate_model_path)

vehicle_model, plate_model = load_yolo_models()

# ✅ Upload Video
uploaded_file = st.file_uploader("📂 Upload a video file", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file:
    # Save uploaded file to temp directory
    temp_dir = tempfile.TemporaryDirectory()
    input_video_path = os.path.join(temp_dir.name, uploaded_file.name)
    
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(input_video_path)  # Show uploaded video
    
    # Process the video
    st.write("⏳ Processing video...")
    output_video_path = os.path.join(temp_dir.name, "output.mp4")
    csv_output_path = os.path.join(temp_dir.name, "speeding_vehicles.csv")

    run_detection(input_video_path, output_video_path, csv_output_path)

    st.success("✅ Processing complete!")

    # ✅ Show Processed Video
    st.subheader("📌 Processed Video Output")
    st.video(output_video_path)

    # ✅ Show CSV Data
    st.subheader("📄 Detected Speed Violations")
    df = pd.read_csv(csv_output_path)
    st.dataframe(df)
'''
import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from main import run_detection  # Import your main detection function

# ✅ Streamlit UI Setup
st.set_page_config(page_title="Vehicle Speed & License Plate Detection", layout="wide")
st.title("🚗 Speeding Vehicle Detection with License Plate Recognition")

# ✅ Hugging Face Model Download
@st.cache_resource
def load_yolo_models():
    plate_model_path = hf_hub_download(repo_id="deepakpro190/my-yolo-plate-detect-model", filename="best.pt")
    return YOLO("yolov8n.pt"), YOLO(plate_model_path)

vehicle_model, plate_model = load_yolo_models()

# ✅ Upload Video
uploaded_file = st.file_uploader("📂 Upload a video file", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file:
    # Save uploaded file to temp directory
    temp_dir = tempfile.TemporaryDirectory()
    input_video_path = os.path.join(temp_dir.name, uploaded_file.name)
    
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(input_video_path)  # Show uploaded video
    
    # Process the video
    st.write("⏳ Processing video... Please wait.")
    output_video_path = os.path.join(temp_dir.name, "output.mp4")
    csv_output_path = os.path.join(temp_dir.name, "speeding_vehicles.csv")

    run_detection(input_video_path, output_video_path, csv_output_path)

    st.success("✅ Processing complete!")

    # ✅ Show Processed Video
    st.subheader("📌 Processed Video Output")
    
    # Ensure the video file exists before displaying
    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
        st.video(output_video_path)
    else:
        st.error("⚠️ Processed video is missing or empty. Check the processing function.")

    # ✅ Download Processed Video
    with open(output_video_path, "rb") as file:
        st.download_button(
            label="📥 Download Processed Video",
            data=file,
            file_name="processed_video.mp4",
            mime="video/mp4",
        )

    # ✅ Show CSV Data
    st.subheader("📄 Detected Speed Violations")
    
    # Ensure CSV is not empty before displaying
    if os.path.exists(csv_output_path) and os.path.getsize(csv_output_path) > 0:
        df = pd.read_csv(csv_output_path)
        st.dataframe(df)
        
        # ✅ Download CSV
        with open(csv_output_path, "rb") as csv_file:
            st.download_button(
                label="⬇️ Download Results",
                data=csv_file,
                file_name="speeding_vehicles.csv",
                mime="text/csv",
            )
    else:
        st.error("⚠️ No data found in CSV. Ensure detection is working properly.")

    # Cleanup temporary files
    temp_dir.cleanup()

    # ✅ Download CSV
    st.download_button("⬇️ Download Results", data=open(csv_output_path, "rb"), file_name="speeding_vehicles.csv", mime="text/csv")

    temp_dir.cleanup()  # Clean up temporary files
