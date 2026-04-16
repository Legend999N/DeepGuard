from src.model import load_model
from src.preprocess import preprocess_image
from src.gradcam import generate_heatmap
import cv2

model = load_model()

img_path = "Dataset/Train/Fake/fake_0.jpg"  # any image

tensor = preprocess_image(img_path)

heatmap = generate_heatmap(model, tensor)

cv2.imshow("Heatmap", heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()