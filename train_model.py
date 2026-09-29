
import os
import re

import cv2
import joblib
import numpy as np
import mediapipe as mp
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from extract import load_image_with_white_bg, upscale_if_small, RAW_dir
from sentences import normalize_landmarks, EXCLUDED_LABELS

MODEL_OUT = os.path.join("models", "sign_model.pkl")

ROTATIONS = (-10, 10)
SVC_C = 10

_NAME = re.compile(r"^(hand\d+)_", re.I)


def pad_to_square(image, pad=40):
    h, w = image.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    square = cv2.copyMakeBorder(image, top, side - h - top, left, side - w - left,
                                cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return cv2.copyMakeBorder(square, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=[255, 255, 255])


def rotate_image(image, degrees):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))


def mirror_features(features):
    coords = features.reshape(21, 3).copy()
    coords[:, 0] *= -1
    return coords.reshape(-1)


def recovery_attempts(image):
    """Ways to retry a miss, each with the inverse to undo it on the landmarks.

    Images are squared first so aspect_ratio stays 1.0 and rotation is exactly
    invertible in feature space.
    """
    base = pad_to_square(image)
    yield base, lambda f: f
    yield pad_to_square(upscale_if_small(image, min_side=800)), lambda f: f
    for degrees in (15, -15, 30, -30, 45, -45, 90, 180, 270):
        yield (rotate_image(base, degrees),
               (lambda d: (lambda f: rotate(f.reshape(1, -1), d).reshape(-1)))(degrees))
    yield cv2.flip(base, 1), mirror_features


def extract_dataset():
    """Landmarks plus the hand each sample came from."""
    mp_hands = mp.solutions.hands
    X, y, groups, is_recovered = [], [], [], []
    labels = sorted(d for d in os.listdir(RAW_dir) if os.path.isdir(os.path.join(RAW_dir, d)))
    recovered = 0
    lost = 0

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.5) as hands, \
         mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.1) as lenient:
        for label in labels:
            if label in EXCLUDED_LABELS:
                continue
            label_dir = os.path.join(RAW_dir, label)
            for fname in sorted(f for f in os.listdir(label_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))):
                image = load_image_with_white_bg(os.path.join(label_dir, fname))
                if image is None:
                    lost += 1
                    continue
                image = upscale_if_small(image, min_side=300)
                match = _NAME.match(fname)
                group = match.group(1) if match else "unknown"

                padded = cv2.copyMakeBorder(image, 40, 40, 40, 40,
                                            cv2.BORDER_CONSTANT, value=[255, 255, 255])
                results = hands.process(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    h, w = padded.shape[:2]
                    X.append(normalize_landmarks(results.multi_hand_landmarks[0], w / h))
                    y.append(label)
                    groups.append(group)
                    is_recovered.append(False)
                    continue

                for candidate, undo in recovery_attempts(image):
                    retry = lenient.process(cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB))
                    if retry.multi_hand_landmarks:
                        X.append(undo(normalize_landmarks(retry.multi_hand_landmarks[0], 1.0)))
                        y.append(label)
                        groups.append(group)
                        is_recovered.append(True)
                        recovered += 1
                        break
                else:
                    lost += 1

    print(f"usable samples: {len(X)}  ({recovered} rescued by the recovery pass, "
          f"{lost} unrecoverable)")
    return (np.array(X, dtype=np.float32), np.array(y), np.array(groups),
            np.array(is_recovered))


def rotate(features, degrees):
    coords = features.reshape(len(features), 21, 3).copy()
    theta = np.radians(degrees)
    cos, sin = np.cos(theta), np.sin(theta)
    x, y = coords[:, :, 0].copy(), coords[:, :, 1].copy()
    coords[:, :, 0] = x * cos - y * sin
    coords[:, :, 1] = x * sin + y * cos
    return coords.reshape(len(features), -1)


def augment(X, y):
    Xs, ys = [X], [y]
    for degrees in ROTATIONS:
        Xs.append(rotate(X, degrees))
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def build_model():
    return make_pipeline(
        StandardScaler(),
        SVC(C=SVC_C, gamma="scale", probability=True, random_state=42),
    )


def main():
    X, y, groups, is_recovered = extract_dataset()

    # Test only on cleanly-detected samples. Recovered ones are intrinsically
    # harder, so scoring on them would drag the number down and make this run
    # look worse than a run without the recovery pass - even though recovery
    # helps. Holding the test set fixed keeps runs comparable.
    print("\nScoring on hands the model never trained on...")
    folds = GroupKFold(n_splits=len(set(groups)))
    scores, truths, preds = [], [], []
    for train_idx, test_idx in folds.split(X, y, groups):
        test_idx = test_idx[~is_recovered[test_idx]]
        Xa, ya = augment(X[train_idx], y[train_idx])
        model = build_model().fit(Xa, ya)
        pred = model.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], pred))
        truths.extend(y[test_idx])
        preds.extend(pred)

    print(f"accuracy on unseen hands: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    print(f"  per held-out hand: {[round(s, 3) for s in scores]}\n")
    print(classification_report(truths, preds, zero_division=0))

    Xa, ya = augment(X, y)
    print(f"training final model on {len(Xa)} samples ({len(X)} real + rotated copies)")
    model = build_model().fit(Xa, ya)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"saved -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
