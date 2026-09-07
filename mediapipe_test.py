import cv2
import json
import mediapipe as mp
import numpy as np
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

from temporal_decoder import TemporalPredictionDecoder


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "asl_baseline_model.keras"

CLASS_NAMES_PATH = "class_names.json"

INFERENCE_CONFIG_PATH = "inference_config.json"

HAND_LANDMARKER_PATH = "hand_landmarker.task"

CROP_DIR = "webcam_crops"

INPUT_SIZE = 224


# ============================================================
# LOAD CLASS NAMES
# ============================================================

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Could not find {CLASS_NAMES_PATH}. "
        "Run classification.py first."
    )


with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


print("\nLoaded classes:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# CREATE CROP DIRECTORY
# ============================================================

os.makedirs(CROP_DIR, exist_ok=True)

existing_numbers = []

for filename in os.listdir(CROP_DIR):

    if filename.startswith("hand_") and filename.endswith(".jpg"):

        try:
            number = int(
                filename[
                    len("hand_") : -len(".jpg")
                ]
            )

            existing_numbers.append(number)

        except ValueError:
            pass


if existing_numbers:
    save_counter = max(existing_numbers) + 1
else:
    save_counter = 1


# ============================================================
# LOAD INFERENCE CONFIG
# ============================================================

print("\nLoading temporal inference configuration...")


if not os.path.exists(INFERENCE_CONFIG_PATH):
    raise FileNotFoundError(
        f"Could not find {INFERENCE_CONFIG_PATH}. "
        "Run classification.py first."
    )


with open(INFERENCE_CONFIG_PATH, "r") as f:
    inference_config = json.load(f)


print("\nTemporal configuration:")

for key, value in inference_config.items():
    print(f"  {key}: {value}")


# ============================================================
# LOAD MEDIAPIPE
# ============================================================

print("\nLoading MediaPipe Hand Landmarker...")


base_options = python.BaseOptions(
    model_asset_path=HAND_LANDMARKER_PATH
)


options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE,
)


detector = vision.HandLandmarker.create_from_options(
    options
)


print("MediaPipe loaded successfully!")


# ============================================================
# LOAD CNN
# ============================================================

print("\nLoading CNN model...")


classifier = load_model(MODEL_PATH)


print("CNN model loaded successfully!")

print(
    f"Model input shape: "
    f"{classifier.input_shape}"
)

print(
    f"Model output shape: "
    f"{classifier.output_shape}"
)


# ============================================================
# VERIFY MODEL
# ============================================================

num_outputs = classifier.output_shape[-1]


if num_outputs != len(class_names):

    raise ValueError(
        "\nMODEL / CLASS MISMATCH\n"
        f"Model outputs: {num_outputs}\n"
        f"Class names: {len(class_names)}\n"
        "Retrain the model and make sure "
        "class_names.json belongs to that model."
    )


# ============================================================
# TEMPORAL DECODER
# ============================================================

decoder = TemporalPredictionDecoder(
    window_size=inference_config.get(
        "window_size",
        15
    ),

    min_stable_predictions=inference_config.get(
        "min_stable_predictions",
        8
    ),

    min_confidence=inference_config.get(
        "min_confidence",
        0.45
    ),

    min_margin=inference_config.get(
        "min_margin",
        0.15
    ),

    min_consensus_ratio=inference_config.get(
        "min_consensus_ratio",
        0.60
    ),

    suppress_duplicate=inference_config.get(
        "suppress_duplicate",
        True
    ),

    recency_weight=inference_config.get(
        "recency_weight",
        1.15
    ),
)


print("\nTemporal decoder initialized.")

print(
    f"Window size: "
    f"{decoder.window_size}"
)

print(
    f"Minimum stable predictions: "
    f"{decoder.min_stable_predictions}"
)

print(
    f"Minimum confidence: "
    f"{decoder.min_confidence:.2f}"
)

print(
    f"Minimum margin: "
    f"{decoder.min_margin:.2f}"
)

print(
    f"Minimum consensus: "
    f"{decoder.min_consensus_ratio:.2f}"
)

print(
    f"Recency weight: "
    f"{decoder.recency_weight:.2f}"
)


# ============================================================
# SENTENCE / WORD STATE
# ============================================================

# Completed words
sentence = ""

# Letters currently being signed
current_word = ""

# Last letter committed to the current word
last_committed_letter = None

# Prevent a continuously-held sign from being repeatedly added
letter_locked = False


# ============================================================
# HAND ABSENCE DETECTION
# ============================================================

# MediaPipe can occasionally lose the hand for a frame or two.
# Therefore, we do not immediately end a word.

no_hand_frames = 0

# Number of consecutive frames without a hand required to
# consider the word finished.
NO_HAND_FRAMES_REQUIRED = 15

# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    detector.close()

    raise SystemExit


print("Webcam opened successfully!")


print(
    """
Controls:

Q / q       -> Quit
S / s       -> Save current hand crop
SPACE       -> Add a space
BACKSPACE   -> Delete last character
C / c       -> Clear sentence

Click the webcam window once so it has keyboard focus.
"""
)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ====================================================
        # READ FRAME
        # ====================================================

        success, frame = cap.read()


        if not success:

            print(
                "ERROR: Could not read frame."
            )

            break


        # ====================================================
        # MIRROR CAMERA
        # ====================================================

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape


        # ====================================================
        # CURRENT CROP
        # ====================================================

        current_hand_crop = None


        # ====================================================
        # BGR -> RGB
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ====================================================
        # DETECT HAND
        # ====================================================

        result = detector.detect(mp_image)


        # ====================================================
        # HAND DETECTED
        # ====================================================

        if result.hand_landmarks:
            # A hand is currently visible, so reset the absence counter.
            no_hand_frames = 0
            hand_landmarks = result.hand_landmarks[0]

            # =================================================
            # LANDMARK COORDINATES
            # =================================================

            xs = [
                landmark.x * w
                for landmark in hand_landmarks
            ]

            ys = [
                landmark.y * h
                for landmark in hand_landmarks
            ]


            min_x = min(xs)
            max_x = max(xs)

            min_y = min(ys)
            max_y = max(ys)


            # =================================================
            # HAND SIZE
            # =================================================

            raw_width = max_x - min_x
            raw_height = max_y - min_y

            hand_size = max(
                raw_width,
                raw_height
            )


            if hand_size <= 1:

                decoder.reset()

                continue


            # =================================================
            # PADDING
            # =================================================

            padding = max(
                int(0.25 * hand_size),
                10
            )


            x_min = int(min_x) - padding
            y_min = int(min_y) - padding

            x_max = int(max_x) + padding
            y_max = int(max_y) + padding


            # =================================================
            # SQUARE CROP
            # =================================================

            box_width = x_max - x_min
            box_height = y_max - y_min

            side = max(
                box_width,
                box_height
            )


            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2


            x_min = center_x - side // 2
            y_min = center_y - side // 2

            x_max = x_min + side
            y_max = y_min + side


            # =================================================
            # CLAMP TO IMAGE
            # =================================================

            if x_min < 0:

                x_max -= x_min
                x_min = 0


            if y_min < 0:

                y_max -= y_min
                y_min = 0


            if x_max > w:

                shift = x_max - w

                x_min -= shift
                x_max = w


            if y_max > h:

                shift = y_max - h

                y_min -= shift
                y_max = h


            x_min = max(0, x_min)
            y_min = max(0, y_min)

            x_max = min(w, x_max)
            y_max = min(h, y_max)


            # =================================================
            # EXTRACT CROP
            # =================================================

            if (
                x_max > x_min
                and y_max > y_min
            ):

                hand_crop = frame[
                    y_min:y_max,
                    x_min:x_max
                ]

            else:

                hand_crop = None


            # =================================================
            # VALID CROP
            # =================================================

            if (
                hand_crop is not None
                and hand_crop.size > 0
            ):

                current_hand_crop = hand_crop.copy()


                # =============================================
                # DRAW BOUNDING BOX
                # =============================================

                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
                )


                # =============================================
                # CNN PREPROCESSING
                # =============================================

                hand_crop_rgb = cv2.cvtColor(
                    current_hand_crop,
                    cv2.COLOR_BGR2RGB
                )

                cnn_input = cv2.resize(
                    hand_crop_rgb,
                    (INPUT_SIZE, INPUT_SIZE),
                    interpolation=cv2.INTER_AREA
                )


                # IMPORTANT:
                #
                # Do NOT divide by 255.
                #
                # EfficientNetB0 in this model
                # contains its own preprocessing.

                cnn_input = cnn_input.astype(
                    np.float32
                )


                cnn_input = np.expand_dims(
                    cnn_input,
                    axis=0
                )


                # =============================================
                # CNN PREDICTION
                # =============================================

                prediction = classifier.predict(
                    cnn_input,
                    verbose=0
                )[0]


                # =============================================
                # TOP 3
                # =============================================

                top_indices = np.argsort(
                    prediction
                )[-3:][::-1]


                pred_index = int(
                    top_indices[0]
                )


                confidence = float(
                    prediction[pred_index]
                )


                predicted_letter = class_names[
                    pred_index
                ]


                top3_text = " | ".join(
                    [
                        (
                            f"{class_names[i]}:"
                            f"{prediction[i]:.2f}"
                        )
                        for i in top_indices
                    ]
                )


                # =============================================
                # TEMPORAL DECODER
                # =============================================
                #
                # IMPORTANT:
                #
                # TemporalPredictionDecoder expects:
                #
                #     update(prediction_label, probabilities)
                #
                # It does NOT expect class_names/config_path.
                #
                # It returns:
                #
                #     accepted_letter
                #     or
                #     None
                #

                decoder_result = decoder.update(
                    predicted_letter,
                    prediction
                )


                # =============================================
                # ACCEPTED CHARACTER
                # =============================================

                if decoder_result is not None:
                    stable_letter = decoder_result

                    # -------------------------------------------------
                    # COMMIT GATE
                    # -------------------------------------------------

                    should_commit = False

                    if not letter_locked:

                        should_commit = True

                    elif stable_letter != last_committed_letter:

                        should_commit = True


                    # -------------------------------------------------
                    # ADD LETTER TO CURRENT WORD
                    # -------------------------------------------------

                    if should_commit:

                        current_word += stable_letter

                        last_committed_letter = stable_letter

                        letter_locked = True


                        print(
                            "\n"
                            + "=" * 50
                        )

                        print(
                            "ACCEPTED LETTER: "
                            f"{stable_letter}"
                        )

                        print(
                            f"Current word: "
                            f"{current_word}"
                        )

                        print(
                            f"Completed sentence: "
                            f"{sentence}"
                        )

                        print(
                            "=" * 50
                        )


                    else:

                        print(
                            f"Duplicate '{stable_letter}' suppressed."
                        )

                # =============================================
                # TEMPORAL DEBUG VALUES
                # =============================================

                history_size = len(
                    decoder.prediction_history
                )


                if (
                    history_size > 0
                    and len(
                        decoder.prediction_history
                    ) >= decoder.min_stable_predictions
                ):

                    from collections import Counter

                    counts = Counter(
                        decoder.prediction_history
                    )

                    _, majority_count = (
                        counts.most_common(1)[0]
                    )

                    consensus = (
                        majority_count
                        / history_size
                    )

                else:

                    consensus = 0.0


                # =============================================
                # DISPLAY RAW PREDICTION
                # =============================================

                prediction_label = (
                    f"{predicted_letter} "
                    f"({confidence:.2f})"
                )


                cv2.putText(
                    frame,
                    prediction_label,
                    (
                        x_min,
                        max(y_min - 10, 30)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )


                # =============================================
                # TEMPORAL STATUS
                # =============================================

                if decoder_result is not None:

                    status_text = (
                        "ACCEPTED: "
                        + decoder_result
                    )

                elif history_size < decoder.min_stable_predictions:

                    status_text = (
                        f"STABILIZING "
                        f"({history_size}/"
                        f"{decoder.min_stable_predictions})"
                    )

                else:

                    status_text = "STABILIZING"

                cv2.putText(
                    frame,
                    status_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


                temporal_text = (
                    f"History: "
                    f"{history_size}/"
                    f"{decoder.window_size}"
                    f" | Consensus: "
                    f"{consensus:.2f}"
                )


                cv2.putText(
                    frame,
                    temporal_text,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )


                # =============================================
                # TOP 3 DISPLAY
                # =============================================

                top3_display = (
                    f"Top: {top3_text}"
                )


                cv2.putText(
                    frame,
                    top3_display,
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )


                # =============================================
                # LANDMARKS
                # =============================================

                for landmark in hand_landmarks:

                    x = int(
                        landmark.x * w
                    )

                    y = int(
                        landmark.y * h
                    )


                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )


                # =============================================
                # CONNECTIONS
                # =============================================

                for start, end in HAND_CONNECTIONS:

                    x1 = int(
                        hand_landmarks[start].x * w
                    )

                    y1 = int(
                        hand_landmarks[start].y * h
                    )

                    x2 = int(
                        hand_landmarks[end].x * w
                    )

                    y2 = int(
                        hand_landmarks[end].y * h
                    )


                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            # ------------------------------------------------
            # Count consecutive frames where no hand is found
            # ------------------------------------------------

            no_hand_frames += 1


            # ------------------------------------------------
            # TEMPORARY HAND LOSS
            # ------------------------------------------------
            #
            # MediaPipe may occasionally lose track of a hand
            # for a few frames. Do not immediately end the word.
            # ------------------------------------------------

            if no_hand_frames < NO_HAND_FRAMES_REQUIRED:

                cv2.putText(
                    frame,
                    (
                        f"Hand lost "
                        f"({no_hand_frames}/"
                        f"{NO_HAND_FRAMES_REQUIRED})"
                    ),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )


            # ------------------------------------------------
            # HAND ABSENT LONG ENOUGH
            # ------------------------------------------------

            elif no_hand_frames == NO_HAND_FRAMES_REQUIRED:

                # Only complete a word if we have actually
                # accepted at least one letter.

                if current_word:

                    # Add a space between completed words.
                    if sentence:

                        sentence += " "

                    # Move the current word into the sentence.
                    sentence += current_word


                    print(
                        "\n"
                        + "=" * 50
                    )

                    print(
                        "WORD COMPLETED: "
                        f"{current_word}"
                    )

                    print(
                        f"Sentence: "
                        f"{sentence}"
                    )

                    print(
                        "=" * 50
                    )


                    # Clear the current word so the next
                    # detected letters form a new word.

                    current_word = ""


                # Reset the temporal decoder.

                decoder.reset()


                # Unlock letter recognition so that the same
                # letter can be used in the next word.

                letter_locked = False

                last_committed_letter = None


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            cv2.putText(
                frame,
                "No hand detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ====================================================
        # SENTENCE DISPLAY
        # ====================================================

        cv2.rectangle(
            frame,
            (10, h - 65),
            (w - 10, h - 10),
            (0, 0, 0),
            -1
        )


        # Combine completed words and the word currently being signed.

        if sentence and current_word:

            display_text = (
                sentence
                + " "
                + current_word
            )

        elif current_word:

            display_text = current_word

        else:

            display_text = sentence


        cv2.putText(
            frame,
            f"Sentence: {display_text}",
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "ASL Recognition",
            frame
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # ====================================================
        # QUIT
        # ====================================================

        if key in (ord("q"), ord("Q")):

            break


        # ====================================================
        # SAVE CROP
        # ====================================================

        elif key in (ord("s"), ord("S")):

            if (
                current_hand_crop is not None
                and current_hand_crop.size > 0
            ):

                filename = os.path.join(
                    CROP_DIR,
                    f"hand_{save_counter:04d}.jpg"
                )


                saved = cv2.imwrite(
                    filename,
                    current_hand_crop
                )


                if saved:

                    print(
                        f"[SAVED] {filename}"
                    )

                    save_counter += 1

                else:

                    print(
                        f"[ERROR] "
                        f"Could not save "
                        f"{filename}"
                    )

            else:

                print(
                    "[NOT SAVED] "
                    "No hand crop available."
                )


        # ====================================================
        # SPACE
        # ====================================================

        elif key == 32:

            # Manually finish the current word.

            if current_word:

                if sentence:

                    sentence += " "

                sentence += current_word


                print(
                    f"Word completed: "
                    f"{current_word}"
                )


                current_word = ""


            decoder.reset()

            letter_locked = False

            last_committed_letter = None

            no_hand_frames = 0


        # ====================================================
        # BACKSPACE
        # ====================================================

        elif key in (8, 127):

            # First delete from the word currently being built.

            if current_word:

                current_word = current_word[:-1]

                print(
                    f"Current word: "
                    f"{current_word}"
                )


            # If there is no current word, delete from the
            # completed sentence.

            elif sentence:

                sentence = sentence[:-1]

                print(
                    f"Sentence: "
                    f"{sentence}"
                )


        # ====================================================
        # CLEAR
        # ====================================================

        elif key in (ord("c"), ord("C")):

            sentence = ""

            current_word = ""

            decoder.reset()

            letter_locked = False

            last_committed_letter = None

            no_hand_frames = 0


            print(
                "Sentence cleared."
            )

finally:

    cap.release()

    cv2.destroyAllWindows()

    try:
        detector.close()
    except Exception:
        pass


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    f"\nFinal sentence: {sentence}"
)

print("Program ended.")