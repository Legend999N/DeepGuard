import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model, save_model, device

# ── Transforms ────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Dataset Loader ────────────────────────────────────────────────────────────
def get_dataloaders(dataset_path, batch_size=8):
    train_path = os.path.join(dataset_path, "Train")
    val_path   = os.path.join(dataset_path, "Validation")

    train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(val_path,   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0)

    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Classes       : {train_dataset.classes}")

    return train_loader, val_loader


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_model(dataset_path="Dataset", epochs=3, batch_size=8, lr=1e-4):

    print("=" * 50)
    print("  DeepGuard — Optimized Training 🚀")
    print("=" * 50)

    train_loader, val_loader = get_dataloaders(dataset_path, batch_size)

    model = build_model(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_acc = 0.0

    for epoch in range(epochs):

        # ── TRAIN ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            if batch_idx > 800:   # 🔥 LIMIT TRAINING
                break

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 20 == 0:
                acc = 100. * train_correct / train_total
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Batch [{batch_idx+1}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {acc:.1f}%")

        train_acc = 100. * train_correct / train_total
        avg_loss = train_loss / (batch_idx + 1)   # ✅ FIXED

        # ── VALIDATION ──
        model.eval()
        val_correct, val_total = 0, 0

        print("\n🔍 Running validation...")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):

                if batch_idx > 200:   # 🔥 LIMIT VALIDATION
                    break

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = outputs.max(1)

                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # ✅ Progress print (no freeze feeling)
                if (batch_idx + 1) % 50 == 0:
                    print(f"Validation Batch {batch_idx+1}")

        val_acc = 100. * val_correct / val_total

        print("\n" + "=" * 50)
        print(f"Epoch {epoch+1}/{epochs} Results")
        print(f"Train Loss : {avg_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Val Acc    : {val_acc:.2f}%")
        print("=" * 50)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, "models/efficientnet_deepfake.pth")
            print(f"✅ Best model saved! Val Acc: {val_acc:.2f}%\n")

    print("🎉 Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_model(
        dataset_path="Dataset",
        epochs=3,
        batch_size=8,
        lr=1e-4
    )