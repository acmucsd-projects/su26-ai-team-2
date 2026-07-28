"""Extract normalized MediaPipe hand landmarks from the ASL dataset."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


DATASET_PATH = Path("data/asl_dataset")
MODEL_PATH = Path("models/hand_landmarker.task")
OUTPUT_PATH = Path("output/landmarks.csv")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def normalize_landmarks(landmarks):
    """Normalize landmarks relative to the wrist and hand size."""
    coordinates = np.array(
        [[point.x, point.y, point.z] for point in landmarks],
        dtype=np.float32
    )

    wrist = coordinates[0]
    coordinates = coordinates - wrist

    distances = np.linalg.norm(coordinates, axis=1)
    hand_size = np.max(distances)

    if hand_size > 0:
        coordinates = coordinates / hand_size

    return coordinates.flatten()


def create_landmarker():
    """Create a MediaPipe Hand Landmarker for individual images."""
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def extract_image_features(image_path, landmarker):
    """Return normalized landmarks from one image."""
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mediapipe_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    result = landmarker.detect(mediapipe_image)

    if not result.hand_landmarks:
        return None

    features = normalize_landmarks(result.hand_landmarks[0])

    handedness = result.handedness[0][0].category_name
    handedness_value = 1 if handedness == "Right" else 0

    return np.append(features, handedness_value)


def main():
    """Extract landmarks from all labeled dataset images."""
    rows = []
    detected_count = 0
    skipped_count = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with create_landmarker() as landmarker:
        for class_directory in sorted(DATASET_PATH.iterdir()):
            if not class_directory.is_dir():
                continue

            label = class_directory.name

            for image_path in class_directory.iterdir():
                if image_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                features = extract_image_features(
                    image_path,
                    landmarker
                )

                if features is None:
                    skipped_count += 1
                    continue

                row = {
                    f"feature_{index}": value
                    for index, value in enumerate(features)
                }

                row["label"] = label
                row["image_path"] = str(image_path)

                rows.append(row)
                detected_count += 1

                if detected_count % 500 == 0:
                    print(f"Processed {detected_count} images")

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {detected_count} samples to {OUTPUT_PATH}")
    print(f"Skipped {skipped_count} images with no detected hand")
    print(f"Dataset shape: {dataframe.shape}")


if __name__ == "__main__":
    main()