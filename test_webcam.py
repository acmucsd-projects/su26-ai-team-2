"""Run real-time ASL letter classification using a webcam."""

from collections import Counter
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


LANDMARK_MODEL_PATH = Path("models/hand_landmarker.task")
CLASSIFIER_PATH = Path("output/asl_classifier.joblib")

CONFIDENCE_THRESHOLD = 0.65
PREDICTION_HISTORY_SIZE = 8


def normalize_landmarks(landmarks):
    """Normalize landmarks relative to the wrist and hand size."""
    coordinates = np.array(
        [[point.x, point.y, point.z] for point in landmarks],
        dtype=np.float32
    )

    coordinates = coordinates - coordinates[0]

    hand_size = np.max(np.linalg.norm(coordinates, axis=1))

    if hand_size > 0:
        coordinates = coordinates / hand_size

    return coordinates.flatten()


def get_stable_prediction(prediction_history):
    """Return the most common recent prediction."""
    if not prediction_history:
        return None

    counts = Counter(prediction_history)
    return counts.most_common(1)[0][0]


def main():
    """Open the webcam and classify detected hand signs."""
    saved_data = joblib.load(CLASSIFIER_PATH)
    classifier = saved_data["model"]

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(LANDMARK_MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    camera = cv2.VideoCapture(0)
    prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)
    frame_number = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:
        while camera.isOpened():
            success, frame = camera.read()

            if not success:
                break

            frame_number += 1
            timestamp_ms = int(
                frame_number * 1000 / 30
            )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mediapipe_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect_for_video(
                mediapipe_image,
                timestamp_ms
            )

            displayed_label = "No hand detected"

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                features = normalize_landmarks(landmarks)

                handedness = (
                    result.handedness[0][0].category_name
                )
                handedness_value = (
                    1 if handedness == "Right" else 0
                )

                features = np.append(
                    features,
                    handedness_value
                ).reshape(1, -1)

                prediction = classifier.predict(features)[0]

                confidence = 1.0

                if hasattr(classifier, "predict_proba"):
                    probabilities = classifier.predict_proba(
                        features
                    )[0]
                    confidence = float(np.max(probabilities))

                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction_history.append(prediction)

                stable_prediction = get_stable_prediction(
                    prediction_history
                )

                if stable_prediction is not None:
                    displayed_label = (
                        f"{stable_prediction} "
                        f"({confidence:.2f})"
                    )

                height, width, _ = frame.shape

                for landmark in landmarks:
                    x_position = int(landmark.x * width)
                    y_position = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x_position, y_position),
                        4,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                displayed_label,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

            cv2.imshow("ASL Classifier", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()