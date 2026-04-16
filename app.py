import streamlit as st
from src.model import load_model, predict_frame
from src.video_utils import extract_frames
from src.preprocess import detect_and_crop_faces
from src.utils import get_device

st.set_page_config(page_title="DeepGuard", layout="wide")

st.title("DeepGuard: Deepfake Detection Demo")
st.write("Upload a video and the app will extract faces and run deepfake detection.")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    with st.spinner("Loading model..."):
        device = get_device()
        model = load_model("models/efficientnet_deepfake.pth", device)

    st.video(uploaded_video)

    with st.spinner("Extracting frames..."):
        frames = extract_frames(uploaded_video, every_n_frames=15, max_frames=10)

    if not frames:
        st.warning("Could not extract frames from the uploaded video.")
    else:
        st.write(f"Analyzing {len(frames)} key frames...")
        results = []
        for idx, frame in enumerate(frames, start=1):
            crops = detect_and_crop_faces(frame)
            if not crops:
                continue
            for crop in crops:
                score, label = predict_frame(crop, model, device)
                results.append((idx, label, score))

        if results:
            for frame_index, label, score in results:
                st.write(f"Frame {frame_index}: **{label}** ({score:.3f})")
        else:
            st.info("No faces were detected in the selected video frames.")
