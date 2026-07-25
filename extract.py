import os 
import csv 
import cv2 
import numpy as np
import mediapipe as mp

RAW_dir = os.path.join("data", "raw")
OUT_dir = os.path.join("data", "process")
OUT_csv = os.path.join(OUT_dir, "landmarks.csv")

NUM_landmarks = 21
NUM_coords = 3
Feature_Count = NUM_landmarks * NUM_coords

def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
    wrists = coords[0].copy()
    coords -= wrists
    max_dist= np.linalg.norm(coords, axis=1).max()
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()

def proccess_datasets():
    mp_hands = mp.solutions.hands
    if not os.path.exists(RAW_dir):
        print(f"Raw data directory not found at {RAW_dir}")
        return
    os.makedirs(OUT_dir, exist_ok=True)
    labels = sorted(d for d in os.listdir(RAW_dir) if os.path.isdir(os.path.join(RAW_dir, d)))
    if not labels:
        print(f"No label directories found in {RAW_dir}")
        return
    print(f"Processing datasets for labels: {labels}")
    rows = []
    skipped = 0
    total = 0

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for label in labels:
            label_dir = os.path.join(RAW_dir, label)
            image_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                print(f"No image files found in {label_dir}")
                continue
            for fname in image_files:
                total += 1
                path = os.path.join(label_dir, fname)
                image = cv2.imread(path)
                if image is None:
                    print(f"Warning: Could not read image {path}. Skipping.")
                    skipped += 1
                    continue
                image = cv2.copyMakeBorder(image, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                if not results.multi_hand_landmarks:
                    print(f"Warning: No hand landmarks detected in {path}. Skipping.")
                    skipped += 1
                    continue
                hand_landmarks = results.multi_hand_landmarks[0]
                features = normalize_landmarks(hand_landmarks)
                rows.append(np.append(features, label))

        header = [f'coord_{i}' for i in range(Feature_Count)] + ['label']
        with open(OUT_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

if __name__ == "__main__":
    proccess_datasets()