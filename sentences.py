import os
import time
import tempfile
import threading
import warnings

os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib
import requests
from playsound import playsound
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.path.join("models", "sign_model.pkl")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.joblib") 
SENTENCES_OUT = os.path.join("data", "process", "sentences.txt")

CONFIDENCE_THRESHOLD = 0.4
EXCLUDED_LABELS = {"j", "z"}
STABLE_FRAMES_REQUIRED = 15
COOLDOWN_SECONDS = 1.0
CAMERA_INDEX = 0

GESTURE_HOLD_FRAMES = 10
GESTURE_COOLDOWN_SECONDS = 1.5
PINCH_RATIO = 0.4

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"

NUM_LANDMARKS = 21
NUM_COORDS = 3
_LAST_TTS_ERROR = {"message": None}

def normalize_landmarks(hand_landmarks, aspect_ratio=1.0):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    coords[:, 0] *= aspect_ratio
    wrist = coords[0].copy()
    coords -= wrist
    max_dist = np.linalg.norm(coords, axis=1).max()
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}"
        )
    model = joblib.load(MODEL_PATH)

    label_encoder = None
    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return model, label_encoder


def predict_letter(model, label_encoder, features):
    features = features.reshape(1, -1)
    if hasattr(model, "feature_names_in_"):
        features = pd.DataFrame(features, columns=model.feature_names_in_)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        allowed = np.array([str(c) not in EXCLUDED_LABELS for c in model.classes_])
        probs = np.where(allowed, probs, 0.0)
        idx = np.argmax(probs)
        confidence = probs[idx]
        raw_label = model.classes_[idx]
    else:
        raw_label = model.predict(features)[0]
        confidence = 1.0
        if str(raw_label) in EXCLUDED_LABELS:
            return None, 0.0

    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence

    if label_encoder is not None and isinstance(raw_label, (int, np.integer)):
        letter = label_encoder.inverse_transform([raw_label])[0]
    else:
        letter = str(raw_label)

    return letter, confidence

def get_last_tts_error():
    return _LAST_TTS_ERROR["message"]


def _tts_fail(message):
    _LAST_TTS_ERROR["message"] = message
    print(message)


def speak_with_elevenlabs(text):
    if not text.strip():
        print("Nothing to speak - sentence is empty.")
        return

    if not ELEVENLABS_API_KEY:
        _tts_fail("ELEVENLABS_API_KEY is not set (add it to your .env file).")
        return

    _LAST_TTS_ERROR["message"] = None

    def _run():
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": ELEVENLABS_MODEL_ID,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            _tts_fail(f"TTS request failed: {e}")
            return

        if response.status_code == 402:
            _tts_fail(f"Voice {ELEVENLABS_VOICE_ID} needs a paid ElevenLabs plan. "
                      "Use a free-tier voice such as pNInz6obpgDQGcFmaJgB (Adam).")
            return

        if response.status_code != 200:
            _tts_fail(f"ElevenLabs API error {response.status_code}: {response.text[:200]}")
            return
        tmp_path = os.path.join(tempfile.gettempdir(), f"sign_tts_{threading.get_ident()}.mp3")
        with open(tmp_path, "wb") as f:
            f.write(response.content)

        try:
            playsound(tmp_path)
        except Exception as e:
            _tts_fail(f"Playback failed: {e}")

    threading.Thread(target=_run, daemon=True).start()
    print(f"Speaking: {text!r}")


def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def is_high_five(hand_landmarks):
    lm = hand_landmarks.landmark
    wrist = lm[0]
    tip_pip_pairs = [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]
    extended = sum(
        1 for tip, pip in tip_pip_pairs if _dist(lm[tip], wrist) > _dist(lm[pip], wrist) * 1.15
    )
    return extended >= 5


def is_quiet_coyote(hand_landmarks):
    lm = hand_landmarks.landmark
    wrist = lm[0]
    scale = _dist(lm[0], lm[9]) or 1e-6
    thumb_to_index = _dist(lm[4], lm[8]) / scale
    thumb_to_middle = _dist(lm[4], lm[12]) / scale
    ring_curled = _dist(lm[16], wrist) < _dist(lm[13], wrist) * 1.05
    pinky_curled = _dist(lm[20], wrist) < _dist(lm[17], wrist) * 1.05
    return thumb_to_index < PINCH_RATIO and thumb_to_middle < PINCH_RATIO and ring_curled and pinky_curled


def classify_gesture(hand_landmarks):
    if is_quiet_coyote(hand_landmarks):
        return "speak"
    if is_high_five(hand_landmarks):
        return "space"
    return None


def main():
    model, label_encoder = load_model()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    sentence = ""
    current_candidate = None
    candidate_streak = 0
    last_committed_letter = None
    last_commit_time = 0.0
    gesture_candidate = None
    gesture_streak = 0
    last_gesture_time = 0.0

    os.makedirs(os.path.dirname(SENTENCES_OUT), exist_ok=True)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:

        print("Press 'q' or ESC to quit, SPACE for space, BACKSPACE to delete, "
              "'c' to clear, 's' to save, 'v' to speak the sentence aloud.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            display_letter = "-"
            display_conf = 0.0
            gesture_label = None
            now = time.time()

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                frame_h, frame_w = frame.shape[:2]
                features = normalize_landmarks(hand_landmarks, frame_w / frame_h)
                letter, confidence = predict_letter(model, label_encoder, features)
                display_conf = confidence

                if letter is not None:
                    display_letter = letter
                    gesture_candidate = None
                    gesture_streak = 0

                    if letter == current_candidate:
                        candidate_streak += 1
                    else:
                        current_candidate = letter
                        candidate_streak = 1

                    can_commit_same = (
                        letter != last_committed_letter
                        or (now - last_commit_time) >= COOLDOWN_SECONDS
                    )

                    if candidate_streak >= STABLE_FRAMES_REQUIRED and can_commit_same:
                        sentence += letter
                        last_committed_letter = letter
                        last_commit_time = now
                        candidate_streak = 0
                else:
                    current_candidate = None
                    candidate_streak = 0

                    gesture_label = classify_gesture(hand_landmarks)
                    if gesture_label == gesture_candidate:
                        gesture_streak += 1
                    else:
                        gesture_candidate = gesture_label
                        gesture_streak = 1

                    gesture_fired = (
                        gesture_label is not None
                        and gesture_streak >= GESTURE_HOLD_FRAMES
                        and now - last_gesture_time >= GESTURE_COOLDOWN_SECONDS
                    )
                    if gesture_fired:
                        if gesture_label == "space":
                            sentence += " "
                            last_committed_letter = None
                        elif gesture_label == "speak":
                            speak_with_elevenlabs(sentence)
                        last_gesture_time = now
                        gesture_streak = 0
            else:
                current_candidate = None
                candidate_streak = 0
                gesture_candidate = None
                gesture_streak = 0

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
            status = f"Sign: {display_letter}  ({display_conf:.2f})"
            if gesture_label:
                status += f"  Gesture: {gesture_label} ({gesture_streak}/{GESTURE_HOLD_FRAMES})"
            cv2.putText(frame, status,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            progress = min(candidate_streak / STABLE_FRAMES_REQUIRED, 1.0)
            bar_w = int(200 * progress)
            cv2.rectangle(frame, (10, 45), (210, 60), (80, 80, 80), 1)
            cv2.rectangle(frame, (10, 45), (10 + bar_w, 60), (0, 200, 0), -1)

            cv2.rectangle(frame, (0, h - 50), (w, h), (30, 30, 30), -1)
            shown = sentence[-40:] if len(sentence) > 40 else sentence
            cv2.putText(frame, shown + "_", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Sign to Sentence", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: 
                break
            elif key == 32:  # space
                sentence += " "
                last_committed_letter = None
            elif key == 8:  # backspace
                sentence = sentence[:-1]
                last_committed_letter = None
            elif key == ord('c'):
                sentence = ""
                last_committed_letter = None
            elif key == ord('s'):
                with open(SENTENCES_OUT, 'a') as f:
                    f.write(sentence + "\n")
                print(f"Saved: {sentence!r} -> {SENTENCES_OUT}")
            elif key == ord('v'):
                speak_with_elevenlabs(sentence)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final sentence: {sentence!r}")


if __name__ == "__main__":
    main()
