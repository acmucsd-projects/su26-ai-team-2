import os
import cv2
import numpy as np

from tensorflow.keras.models import load_model

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("asl_baseline_model.keras")

print("Model loaded successfully!")

# -----------------------------
# Dataset path
# -----------------------------
dataset_path = os.path.expanduser(
    "~/Documents/Kaggle/asl_dataset/asl_dataset"
)

# -----------------------------
# Choose ONE image
# -----------------------------
image_path = os.path.join(
    dataset_path,
    "a",
    "hand1_a_bot_seg_1_cropped.jpeg"
)

print("Testing image:", image_path)

# -----------------------------
# Load image
# -----------------------------
image = cv2.imread(image_path)

if image is None:
    print("Could not load image.")
    exit()

# OpenCV loads images in BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize exactly like training
image = cv2.resize(image, (128, 128))

# Normalize
image = image.astype(np.float32)
image = image / 255.0

# Add batch dimension
image = np.expand_dims(image, axis=0)

# -----------------------------
# Predict
# -----------------------------
prediction = model.predict(image, verbose=0)

pred_index = np.argmax(prediction)

confidence = prediction[0][pred_index]

class_names = [
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','c','d','e','f','g','h','i','j',
    'k','l','m','n','o','p','q','r','s','t',
    'u','v','w','x','y','z'
]

print("\nPrediction")
print("----------")
print("Predicted class :", class_names[pred_index])
print("Confidence      :", confidence)
print("Actual class    : a")