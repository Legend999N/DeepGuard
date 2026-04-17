from src.model import load_model, predict
from src.preprocess import preprocess_image
from src.gradcam import generate_heatmap
import cv2

# Load trained model
model = load_model("models/efficientnet_deepfake.pth")

# 🔥 Change this path to test images
img_path = "Dataset/Train/Real/real_100.jpg"   # try Fake first

# Preprocess
tensor = preprocess_image(img_path)

if tensor is None:
    print("❌ No face detected")
    exit()

# 🔥 Get prediction
label, confidence, _ = predict(model, tensor)

print("\n==============================")
print(f"Image: {img_path}")
print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2f}%")
print("==============================\n")

# Generate heatmap
heatmap = generate_heatmap(model, tensor)

# Load original image
original = cv2.imread(img_path)
original = cv2.resize(original, (224, 224))

# Convert heatmap to color
heatmap = cv2.applyColorMap((heatmap * 255).astype('uint8'), cv2.COLORMAP_JET)

# Overlay
overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# Show
cv2.imshow("Heatmap", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()