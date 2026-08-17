"""
One-time offline step: runs MediaPipe Hands over every training image and
saves normalized 21-landmark feature vectors to a CSV. Run this once
whenever data/raw changes; training itself then reads straight from the CSV.

Run from the project root (the folder containing data/raw), e.g.:
    python -m landmark_model.extract_landmarks
"""
import os
import csv
import cv2
import mediapipe as mp
import numpy as np

DATA_ROOT = "data/raw"
OUTPUT_CSV = "landmarks.csv"
CLASSES_JSON = "landmark_classes.json"

# Letters that involve motion and can't be represented by a single static frame
EXCLUDED_CLASSES = {"J", "Z"}

mpHands = mp.solutions.hands
# static_image_mode=True because we're processing independent photos, not video
hands = mpHands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


def normalize_landmarks(landmark_list):
    """21 (x, y, z) landmarks -> a 63-dim vector that's translation- and
    scale-invariant, so it doesn't matter where the hand sits in frame or
    how large it appears (unlike raw pixels)."""
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark], dtype=np.float32
    )
    wrist = coords[0].copy()
    coords -= wrist  # translate: wrist becomes the origin

    scale = np.linalg.norm(coords, axis=1).max()  # farthest landmark from wrist
    if scale > 1e-6:
        coords /= scale  # scale: farthest landmark now has distance 1

    return coords.flatten()


def main():
    import json

    classes = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d not in EXCLUDED_CLASSES
    )

    rows_written = 0
    skipped_no_hand = 0

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(63)])

        for label in classes:
            class_dir = os.path.join(DATA_ROOT, label)
            filenames = os.listdir(class_dir)
            print(f"Processing class '{label}': {len(filenames)} images")

            for fname in filenames:
                fpath = os.path.join(class_dir, fname)
                image = cv2.imread(fpath)
                if image is None:
                    continue

                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not results.multi_hand_landmarks:
                    skipped_no_hand += 1
                    continue

                features = normalize_landmarks(results.multi_hand_landmarks[0])
                writer.writerow([label] + features.tolist())
                rows_written += 1

    with open(CLASSES_JSON, "w") as f:
        json.dump(classes, f)

    print(f"\nDone. Wrote {rows_written} rows to {OUTPUT_CSV}")
    print(f"Skipped {skipped_no_hand} images where MediaPipe found no hand")
    print(f"Classes ({len(classes)}): {classes}")


if __name__ == "__main__":
    main()