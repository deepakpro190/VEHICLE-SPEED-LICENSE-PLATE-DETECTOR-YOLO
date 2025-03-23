'''import streamlit as st
import os
import time
import tempfile
import t2  # Importing the function from t2.py

# Set Streamlit page configuration
st.set_page_config(page_title="License Plate Tracker", layout="centered")

# Title
st.title("🚗 License Plate Tracking System")

# File uploader for input video
uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv", "webm"])

# Check if a file has been uploaded
if uploaded_file is not None:
    st.video(uploaded_file)  # Display the uploaded video

    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
        temp_file.write(uploaded_file.read())
        input_video_path = temp_file.name  # Path of the uploaded video

    # Define output video path
    output_video_path = "output/processed_video.mp4"

    # Process button
    if st.button("🚀 Process Video"):
        st.markdown("### Processing Video... Please wait ⏳")
        progress_bar = st.progress(0)

        # Simulate progress update
        for i in range(1, 101):
            time.sleep(0.05)  # Simulate processing delay
            progress_bar.progress(i)

        # Run the model function
        t2.run_license_plate_tracking(input_video_path, output_video_path)

        # Display success message
        st.success("✅ Processing Complete! Download or watch the video below.")

        # Show processed video
        st.video(output_video_path)

        # Provide a download link for the processed video
        with open(output_video_path, "rb") as file:
            st.download_button("📥 Download Processed Video", file, "processed_video.mp4", "video/mp4")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")
'''
import streamlit as st
import time
import os
import cv2
import tempfile
from t2 import run_license_plate_tracking  # Import the function

st.title("🔍 License Plate Tracking App")

# File uploader for input video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv", "webm"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input.close()

    # Define output file path
    output_path = os.path.join(tempfile.gettempdir(), "output.mp4")

    # Show input video preview
    st.video(temp_input.name)

    # Start Processing Button
    if st.button("Start Processing 🚀"):
        st.write("Processing started... Please wait.")

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Get video frame count for accurate progress tracking
        cap = cv2.VideoCapture(temp_input.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        def progress_callback(current_frame):
            """Callback function to update progress bar."""
            progress_percent = int((current_frame / total_frames) * 100)
            progress_bar.progress(progress_percent)
            status_text.write(f"Processing Frame {current_frame} / {total_frames}")

        # Run the video processing function
        run_license_plate_tracking(temp_input.name, output_path, progress_callback)

        # Processing complete
        st.success("Processing Complete ✅")
        progress_bar.progress(100)
        status_text.write("Done!")

        # Show output video
        st.video(output_path)

        # Provide download button for the output video
        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4",
            )

