import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import os

# Initialize MTCNN face detector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    min_face_size=40,
    device=device,
    keep_all=False
)

def detect_and_crop_face(image_input):
    """
    Takes an image path or numpy array
    Returns cropped face as 224x224 tensor or None if no face found
    """
    # Handle both file path and numpy array input
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    else:
        img = image_input

    # Detect and crop face using MTCNN
    face_tensor = mtcnn(img)

    if face_tensor is None:
        return None

    return face_tensor


import torchvision.transforms as transforms

# Same normalization as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(image_input):
    face = detect_and_crop_face(image_input)

    if face is None:
        return None

    # face is already a tensor [3, 224, 224]
    face = transform(face)

    # Add batch dimension
    face = face.unsqueeze(0)

    return face


def extract_frames_from_video(video_path, every_n_frames=10):
    """
    Extracts frames from a video file
    Returns list of frames as numpy arrays
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return frames

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % every_n_frames == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {len(frames)} frames from video")
    return frames


def preprocess_dataset_sample(dataset_path, num_samples=5):
    """
    Test function — processes a few images from dataset
    and prints results so we know everything works
    """
    categories = ['Real', 'Fake']

    for category in categories:
        folder = os.path.join(dataset_path, category)
        images = os.listdir(folder)[:num_samples]

        print(f"\nProcessing {category} images:")
        print("-" * 40)

        for img_name in images:
            img_path = os.path.join(folder, img_name)
            result = preprocess_image(img_path)

            if result is not None:
                print(f"✅ {img_name} → Shape: {result.shape}")
            else:
                print(f"❌ {img_name} → No face detected")


# Run test when this file is executed directly
if __name__ == "__main__":
    dataset_train_path = os.path.join("Dataset", "Train")
    print("🔍 Testing Preprocessing Pipeline...")
    print("=" * 40)
    preprocess_dataset_sample(dataset_train_path)
    print("\n✅ Preprocessing test complete!")
