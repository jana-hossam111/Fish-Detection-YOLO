import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Fish Detection")

st.title("🐟 Fish Detection using YOLOv8")
st.write("Upload an image to detect fish.")

model = YOLO("best.pt")

def warmup_model(model):
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy)

warmup_model(model)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect"):
        results = model(image)

        annotated_image = results[0].plot()

        st.image(
            annotated_image,
            caption="Detection Result",
            use_container_width=True
        )