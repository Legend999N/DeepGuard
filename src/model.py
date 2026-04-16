import torch
import torch.nn as nn
from torchvision import models

# ── Device setup ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def build_model(num_classes=2):
    """
    Loads pretrained EfficientNet-B0 and modifies
    it for binary classification (Real vs Fake)
    """

    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    print("Loaded pretrained EfficientNet-B0")

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    # Unfreeze only classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model.to(device)


def predict(model, face_tensor):
    """
    Takes a preprocessed face tensor [1, 3, 224, 224]
    Returns predicted label and confidence score
    """
    model.eval()

    with torch.no_grad():
        face_tensor = face_tensor.to(device)
        outputs = model(face_tensor)

        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        label = "REAL" if predicted.item() == 0 else "FAKE"
        confidence_pct = confidence.item() * 100

    return label, confidence_pct, probabilities


def save_model(model, path="models/efficientnet_deepfake.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path="models/efficientnet_deepfake.pth"):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model


# ── Quick test when run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.preprocess import preprocess_image

    print("=" * 45)
    print("  Loading EfficientNet-B0 pretrained model...")
    print("=" * 45)

    model = build_model()

    print(f"\n✅ Model loaded successfully!")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test on sample images
    test_images = [
        ("Dataset/Train/Real", "REAL"),
        ("Dataset/Train/Fake", "FAKE"),
    ]

    print("\n" + "=" * 45)
    print("  Running inference on sample images...")
    print("=" * 45)

    for folder, true_label in test_images:
        images = os.listdir(folder)[:3]
        print(f"\nFolder: {folder}")
        print("-" * 45)

        for img_name in images:
            img_path = os.path.join(folder, img_name)
            tensor = preprocess_image(img_path)

            if tensor is None:
                print(f"❌ {img_name} → No face detected")
                continue

            label, confidence, _ = predict(model, tensor)
            status = "✅" if label == true_label else "⚠️"

            print(f"{status} {img_name}")
            print(f"   Predicted: {label}  |  Confidence: {confidence:.1f}%")
            print(f"   True label: {true_label}")

    print("\n" + "=" * 45)
    print("✅ Model ready!")
    print("=" * 45)