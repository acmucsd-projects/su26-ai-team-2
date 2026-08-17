
Mediapipemodel · PY
import json
import cv2
import mediapipe as mp
import torch
from torchvision import transforms
 
from baseline_model.model import ASLCNN
 
MODEL_PATH = "asl_cnn_model.pth"
CLASSES_PATH = "asl_classes.json"
PADDING = 30          # extra pixels around the hand bounding box
CONF_THRESHOLD = 0.6  # only show a prediction above this confidence
 
# --- Load class labels saved during training (J and Z were excluded) ---
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)
 
# --- Load the trained model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLCNN(num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
 
# Same preprocessing pipeline used at training time
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
 
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands
 
# Webcam setup
cap = cv2.VideoCapture(0)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
hands = handsModule.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
 
 
def get_hand_bbox(handLandmarks, frameWidth, frameHeight, padding=PADDING):
    """Pixel-space bounding box around all 21 hand landmarks, with padding."""
    xs = [lm.x * frameWidth for lm in handLandmarks.landmark]
    ys = [lm.y * frameHeight for lm in handLandmarks.landmark]
    x_min = max(int(min(xs)) - padding, 0)
    x_max = min(int(max(xs)) + padding, frameWidth)
    y_min = max(int(min(ys)) - padding, 0)
    y_max = min(int(max(ys)) + padding, frameHeight)
    return x_min, y_min, x_max, y_max
 
 
def predict_letter(hand_crop_bgr):
    """Runs the trained CNN on a cropped hand region, returns (letter, confidence)."""
    if hand_crop_bgr.size == 0:
        return None, 0.0
 
    hand_crop_rgb = cv2.cvtColor(hand_crop_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = transform(hand_crop_rgb).unsqueeze(0).to(device)
 
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)
 
    return classes[predicted_idx.item()], confidence.item()
 
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 
    if results.multi_hand_landmarks is not None:
        for handLandmarks in results.multi_hand_landmarks:
            drawingModule.draw_landmarks(
                frame, handLandmarks, handsModule.HAND_CONNECTIONS
            )
 
            x_min, y_min, x_max, y_max = get_hand_bbox(
                handLandmarks, frameWidth, frameHeight
            )
            hand_crop = frame[y_min:y_max, x_min:x_max]
 
            letter, confidence = predict_letter(hand_crop)
 
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
 
            if letter is not None and confidence >= CONF_THRESHOLD:
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
 
