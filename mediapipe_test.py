import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tensorflow.keras.models import load_model

from collections import deque, Counter
# ----------------------------
# Hand landmark connections
# ----------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

class_names = [
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','c','d','e','f','g','h','i','j',
    'k','l','m','n','o','p','q','r','s','t',
    'u','v','w','x','y','z'
]
prediction_buffer = deque(maxlen=10)

# ----------------------------
# Load MediaPipe Hand Landmarker
# ----------------------------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)
classifier = load_model("asl_baseline_model.keras")

print("CNN model loaded successfully!")

# ----------------------------
# Open webcam
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully!")
print("Press 'q' to quit.")

# ----------------------------
# Main loop
# ----------------------------
while True:
    success, frame = cap.read()

    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    if len(result.hand_landmarks) > 0:
        for hand_landmarks in result.hand_landmarks:
            h, w, _ = frame.shape

            xs = [landmark.x * w for landmark in hand_landmarks]
            ys = [landmark.y * h for landmark in hand_landmarks]

            padding = 0

            x_min = max(int(min(xs)) - padding, 0)
            y_min = max(int(min(ys)) - padding, 0)
            x_max = min(int(max(xs)) + padding, w)
            y_max = min(int(max(ys)) + padding, h)

            # Make the bounding box square
            box_width = x_max - x_min
            box_height = y_max - y_min
            side = max(box_width, box_height)

            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            x_min = max(center_x - side // 2, 0)
            y_min = max(center_y - side // 2, 0)

            x_max = min(x_min + side, w)
            y_max = min(y_min + side, h)

            # If box hits image boundary, shift it back
            if x_max == w:
                x_min = max(w - side, 0)
            if y_max == h:
                y_min = max(h - side, 0)

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size == 0:
                continue

            # Convert crop to RGB for the model
            hand_crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)

            cv2.imshow("Hand Crop", hand_crop)

            # Resize to the same size used during training
            cnn_input = cv2.resize(hand_crop_rgb, (128, 128))

            # Convert to float
            cnn_input = cnn_input.astype(np.float32)

            # Normalize
            cnn_input = cnn_input / 255.0

            # Add batch dimension
            cnn_input = np.expand_dims(cnn_input, axis=0)

            prediction = classifier.predict(cnn_input, verbose=0)

            pred_index = np.argmax(prediction)
            confidence = prediction[0][pred_index]
            predicted_letter = class_names[pred_index]

            prediction_buffer.append(predicted_letter)
            stable_letter = Counter(prediction_buffer).most_common(1)[0][0]


            label = f"{stable_letter} ({confidence:.2f})"
    

            cv2.putText(
                frame,
                label,
                (x_min, max(y_min - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            print(label)

            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()