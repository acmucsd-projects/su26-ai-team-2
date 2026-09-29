import os
import time
import cv2
import mediapipe as mp
import streamlit as st

from sentences import (
    CAMERA_INDEX,
    COOLDOWN_SECONDS,
    SENTENCES_OUT,
    STABLE_FRAMES_REQUIRED,
    load_model,
    normalize_landmarks,
    predict_letter,
    speak_with_elevenlabs,
    get_last_tts_error,
)

GESTURE_HOLD_FRAMES = 10
GESTURE_COOLDOWN_SECONDS = 1.5
PINCH_RATIO = 0.4

st.set_page_config(page_title="Sign to Sentence", layout="wide")


def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def is_high_five(hand_landmarks):
    lm = hand_landmarks.landmark
    wrist = lm[0]
    scale = _dist(lm[0], lm[9]) or 1e-6

    tip_pip_pairs = [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]
    extended = sum(
        1 for tip, pip in tip_pip_pairs if _dist(lm[tip], wrist) > _dist(lm[pip], wrist) * 1.15
    )
    if extended < 5:
        return False

    if _dist(lm[4], lm[13]) / scale < 0.6:
        return False

    return (
        _dist(lm[8], lm[12]) / scale > 0.25
        and _dist(lm[12], lm[16]) / scale > 0.22
        and _dist(lm[16], lm[20]) / scale > 0.22
    )


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


@st.cache_resource
def get_model():
    return load_model()


def main():
    model, label_encoder = get_model()

    left, right = st.columns([3, 1])

    with right:
        st.header("Instructions")
        st.markdown(
            """
            **Sign a Letter or Number**
            Hold a hand sign steady and wait for the progress bar to fill. As of now J and Z are excluded because they require motion to sign.

            **Space** ✋
            Hold up an open hand, all five fingers spread,
            to add a space.

            **Speak**
            Pinch your thumb to your index and middle fingertips with your
            ring and pinky out to have the sentence spoken aloud.
            """
        )
        st.divider()
        sentence_placeholder = st.empty()
        st.divider()
        st.subheader("Signs")
        st.image("https://www.wikihow.com/images/4/4f/Fingerspell-the-Alphabet-in-American-Sign-Language-Summary-Version-2.jpg")

    with left:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    sentence = ""
    current_candidate = None
    candidate_streak = 0
    last_committed_letter = None
    last_commit_time = 0.0
    gesture_candidate = None
    gesture_streak = 0
    last_gesture_time = 0.0

    os.makedirs(os.path.dirname(SENTENCES_OUT), exist_ok=True)

    def render_sentence():
        shown = sentence or "Sign To Start!"
        sentence_placeholder.markdown(f"### Sentence\n```\n{shown}\n```")

    render_sentence()

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.4,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                st.error("Warning: failed to read frame from webcam.")
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

                gesture_label = classify_gesture(hand_landmarks)

                if gesture_label is not None:
                    current_candidate = None
                    candidate_streak = 0

                    if gesture_label == gesture_candidate:
                        gesture_streak += 1
                    else:
                        gesture_candidate = gesture_label
                        gesture_streak = 1

                    if (gesture_streak >= GESTURE_HOLD_FRAMES
                            and now - last_gesture_time >= GESTURE_COOLDOWN_SECONDS):
                        if gesture_label == "space":
                            sentence += " "
                            last_committed_letter = None
                            render_sentence()
                        elif gesture_label == "speak":
                            speak_with_elevenlabs(sentence)
                        last_gesture_time = now
                        gesture_streak = 0
                else:
                    gesture_candidate = None
                    gesture_streak = 0

                    frame_h, frame_w = frame.shape[:2]
                    features = normalize_landmarks(hand_landmarks, frame_w / frame_h)
                    letter, confidence = predict_letter(model, label_encoder, features)
                    display_conf = confidence

                    if letter is not None:
                        display_letter = letter

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
                            render_sentence()
                    else:
                        current_candidate = None
                        candidate_streak = 0
            else:
                current_candidate = None
                candidate_streak = 0
                gesture_candidate = None
                gesture_streak = 0

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            status = f"Sign: {display_letter}  ({display_conf:.2f})"
            if gesture_label:
                status += f"  |  Gesture: {gesture_label} ({gesture_streak}/{GESTURE_HOLD_FRAMES})"
            tts_error = get_last_tts_error()
            if tts_error:
                status += f"  |  SPEAK FAILED: {tts_error}"
            status_placeholder.text(status)

    cap.release()


if __name__ == "__main__":
    main()
