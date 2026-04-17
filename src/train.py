import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import sys
import random

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model, save_model, device


# ── Transforms ──
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


# ── Dataset Loader ──
def get_dataloaders(dataset_path, batch_size=8):

    train_path = os.path.join(dataset_path, "Train")
    val_path   = os.path.join(dataset_path, "Validation")

    train_dataset_full = datasets.ImageFolder(train_path, transform=train_transforms)
    val_dataset        = datasets.ImageFolder(val_path, transform=val_transforms)

    # 🔥 SAFE SUBSET (won’t crash if dataset smaller)
    subset_size = min(18000, len(train_dataset_full))
    indices = random.sample(range(len(train_dataset_full)), subset_size)
    train_dataset = Subset(train_dataset_full, indices)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")

    return train_loader, val_loader


# ── Training Loop ──
def train_model(dataset_path="Dataset", epochs=8, batch_size=8, lr=3e-5):

    print("=" * 50)
    print("🚀 DeepGuard Training (Improved Version)")
    print("=" * 50)

    train_loader, val_loader = get_dataloaders(dataset_path, batch_size)

    model = build_model(num_classes=2)

    criterion = nn.CrossEntropyLoss()

    # 🔥 Train only unfrozen layers
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_val_acc = 0.0

    for epoch in range(epochs):

        # ── TRAIN ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        print(f"\n📚 Epoch {epoch+1}/{epochs} Training...")

        for batch_idx, (images, labels) in enumerate(train_loader):

            # 🔥 Balanced limit (better learning than 600)
            if batch_idx > 1000:
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
                print(f"Batch [{batch_idx+1}] Loss: {loss.item():.4f} Acc: {acc:.1f}%")

        train_acc = 100. * train_correct / train_total
        avg_loss = train_loss / (batch_idx + 1)


        # ── VALIDATION ──
        model.eval()
        val_correct, val_total = 0, 0

        print("\n🔍 Running validation...")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):

                if batch_idx > 200:
                    break

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = outputs.max(1)

                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                if (batch_idx + 1) % 50 == 0:
                    print(f"Validation Batch {batch_idx+1}")

        val_acc = 100. * val_correct / val_total


        # ── RESULTS ──
        print("\n" + "=" * 50)
        print(f"Epoch {epoch+1}/{epochs} Results")
        print(f"Train Loss : {avg_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Val Acc    : {val_acc:.2f}%")
        print("=" * 50)


        # ── SAVE BEST MODEL ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, "models/efficientnet_deepfake.pth")
            print(f"✅ Best model saved! Val Acc: {val_acc:.2f}%\n")

    print("\n🎉 Training Complete!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")


# ── RUN ──
if __name__ == "__main__":
    train_model(
        dataset_path="Dataset",
        epochs=12,
        batch_size=8,
        lr=3e-5
    )