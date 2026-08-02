import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

st.set_page_config(page_title="Fish Detection")

st.title("🐟 Fish Detection using YOLOv8")
st.write("Upload an image to detect fish.")

model = YOLO("best.pt")

def warmup_model(model):
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy)

warmup_model(model) 
frame_count = 0

source = st.radio(
    "Select Input Source",
    ["Image", "Video", "Live Stream"]
)


if source == "Image":

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Detect"):

            results = model(image)

            annotated_image = results[0].plot()

            st.image(
                annotated_image,
                caption="Detection Result",
                use_container_width=True
            )

elif source == "Video":

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:

            temp_video.write(uploaded_video.read())
            video_path = temp_video.name

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)


        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
          "live_detection.mp4",
           fourcc,
            fps,
            (width, height)
)

        video_placeholder = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break
            frame_count += 1

            if frame_count % 2 != 0:
              continue
            frame = cv2.resize(frame, (640, 640))

            results = model(frame)

            annotated_frame = results[0].plot()
            
            out.write(annotated_frame)

            video_placeholder.image(
                annotated_frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()
        out.release()
        st.success("Detection completed!")

        st.video("output.mp4")
        with open("output.mp4", "rb") as file:

         st.download_button(
           label="⬇ Download Detection Video",
           data=file,
           file_name="fish_detection.mp4",
           mime="video/mp4"
    )


elif source == "Live Stream":

    stream_url = st.text_input(
        "Stream URL",
        "http://"
    )

    if st.button("Connect"):

        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            st.error("Cannot open stream")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            fps = 20

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            "live_detection.mp4",
            fourcc,
            fps,
            (width, height)
        )

        placeholder = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            results = model(frame)

            annotated = results[0].plot()

            out.write(annotated)

            placeholder.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        cap.release()
        out.release()

        st.success("Recording Saved!")

        st.video("live_detection.mp4")

        with open("live_detection.mp4", "rb") as file:
            st.download_button(
                "⬇ Download Detection Video",
                data=file,
                file_name="live_detection.mp4",
                mime="video/mp4"
            )