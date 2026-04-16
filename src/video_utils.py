import cv2


def extract_frames(video_file, every_n_frames=15, max_frames=10):
    frames = []
    if hasattr(video_file, "read"):
        data = video_file.read()
        with open("temp_upload_video.mp4", "wb") as temp_file:
            temp_file.write(data)
        video_path = "temp_upload_video.mp4"
    else:
        video_path = video_file

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    frame_index = 0
    saved_frames = 0
    while saved_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % every_n_frames == 0:
            frames.append(frame)
            saved_frames += 1
        frame_index += 1

    cap.release()
    return frames
