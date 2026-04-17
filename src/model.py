import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# ── Device setup ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ── Build Model ───────────────────────────────────────────────────────────────
def build_model(num_classes=2):
    """
    Load EfficientNet-B0 pretrained model and modify for binary classification
    """

    model = EfficientNet.from_pretrained('efficientnet-b0')

    # Replace final layer
    in_features = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )

    # Freeze early layers only
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 blocks (not 2)
    for param in model._blocks[-3:].parameters():
        param.requires_grad = True

    # Unfreeze classifier
    for param in model._fc.parameters():
        param.requires_grad = True

    model = model.to(device)
    return model


# ── Prediction Function ────────────────────────────────────────────────────────
def predict(model, face_tensor):
    """
    Input: face tensor [1, 3, 224, 224]
    Output: label + confidence + probabilities
    """
    model.eval()

    with torch.no_grad():
        face_tensor = face_tensor.to(device)
        outputs = model(face_tensor)

        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        # 🔥 IMPORTANT FIX (label mapping)
        label = "FAKE" if predicted.item() == 0 else "REAL"
        confidence_pct = confidence.item() * 100

    return label, confidence_pct, probabilities


# ── Save Model ─────────────────────────────────────────────────────────────────
def save_model(model, path="models/efficientnet_deepfake.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# ── Load Model ─────────────────────────────────────────────────────────────────
def load_model(path="models/efficientnet_deepfake.pth"):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model


# ── Test Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.preprocess import preprocess_image

    print("=" * 45)
    print("  Loading EfficientNet-B0 model...")
    print("=" * 45)

    model = build_model()

    print(f"\n✅ Model loaded successfully!")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test on sample images
    test_folders = [
        ("Dataset/Train/Real", "REAL"),
        ("Dataset/Train/Fake", "FAKE"),
    ]

    print("\n" + "=" * 45)
    print("  Running inference on sample images...")
    print("=" * 45)

    for folder, true_label in test_folders:
        images = os.listdir(folder)[:3]

        print(f"\nFolder: {folder}")
        print("-" * 45)

        for img_name in images:
            img_path = os.path.join(folder, img_name)

            tensor = preprocess_image(img_path)
            if tensor is None:
                print(f"❌ {img_name} → No face detected")
                continue

            label, confidence, probs = predict(model, tensor)

            status = "✅" if label == true_label else "⚠️"
            print(f"{status} {img_name}")
            print(f"   Predicted: {label} | Confidence: {confidence:.1f}%")
            print(f"   True label: {true_label}")
            print(f"   Raw probs: {probs}")

    print("\n" + "=" * 45)
    print("✅ Model check complete!")
    print("=" * 45)