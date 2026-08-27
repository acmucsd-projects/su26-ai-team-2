"""
Run from the project root:
    python -m mediapipe_model.mediapipeModel
"""
import json
import cv2
import mediapipe as mp
import numpy as np
import torch

from landmark_model.landmark_model import ASLLandmarkMLP

MODEL_PATH = "asl_landmark_model.pth"
CLASSES_PATH = "asl_landmark_classes.json"
CONF_THRESHOLD = 0.6  # only show a prediction above this confidence

# --- Load class labels saved during training (J and Z were excluded) ---
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

# --- Load the trained landmark model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLLandmarkMLP(num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def normalize_landmarks(landmark_list):
    """Must exactly match the normalization used in extract_landmarks.py,
    or the live features won't line up with what the model was trained on."""
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark], dtype=np.float32
    )
    wrist = coords[0].copy()
    coords -= wrist

    scale = np.linalg.norm(coords, axis=1).max()
    if scale > 1e-6:
        coords /= scale

    return coords.flatten()


def predict_letter(landmark_list):
    features = normalize_landmarks(landmark_list)
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)

    return classes[predicted_idx.item()], confidence.item()


drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

cap = cv2.VideoCapture(0)

hands = handsModule.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks is not None:
        for handLandmarks in results.multi_hand_landmarks:
            drawingModule.draw_landmarks(
                frame, handLandmarks, handsModule.HAND_CONNECTIONS
            )

            letter, confidence = predict_letter(handLandmarks)

            xs = [lm.x * w for lm in handLandmarks.landmark]
            ys = [lm.y * h for lm in handLandmarks.landmark]
            x_min, y_min = int(min(xs)), int(min(ys))

            if confidence >= CONF_THRESHOLD:
                label = f"{letter} ({confidence*100:.1f}%)"
                color = (0, 255, 0)
            else:
                label = "..."
                color = (0, 165, 255)

            cv2.putText(
                frame, label, (x_min, max(y_min - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )

    cv2.imshow('ASL Detection', frame)
    if cv2.waitKey(1) == 27:  # Esc to quit
        break

cv2.destroyAllWindows()
cap.release()