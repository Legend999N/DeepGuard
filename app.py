import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.model import load_model, predict
from src.preprocess import preprocess_image
from src.gradcam import generate_heatmap

# ── Load model once ─────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return load_model()

model = load_my_model()

# ── UI ──────────────────────────────────────────────────────
st.set_page_config(page_title="DeepGuard", layout="centered")

st.title("🛡️ DeepGuard — Deepfake Detection")
st.write("Upload an image to detect if it's REAL or FAKE")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ── Process Image ───────────────────────────────────────────
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    tensor = preprocess_image(image)

    if tensor is None:
        st.error("❌ No face detected")
    else:
        # Prediction
        label, confidence, _ = predict(model, tensor)

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Heatmap
        heatmap = generate_heatmap(model, tensor)

        st.subheader("🔥 Heatmap (Model Attention)")
        st.image(heatmap, caption="Grad-CAM Output", use_container_width=True)