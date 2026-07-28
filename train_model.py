"""Train and compare classifiers using extracted hand landmarks."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight


DATA_PATH = Path("output/landmarks.csv")
MODEL_OUTPUT_PATH = Path("output/asl_classifier.joblib")
RESULTS_OUTPUT_PATH = Path("output/model_results.txt")

TEST_SIZE = 0.2
RANDOM_STATE = 42

# MediaPipe hand landmark indices.
FINGERTIPS = [4, 8, 12, 16, 20]
KNUCKLES = [2, 5, 9, 13, 17]


def engineer_features(raw_features):
    """Add fingertip-distance features on top of the 21 raw (x, y, z)
    landmarks + handedness flag.

    These give the model direct signal about finger spread and
    curl (how open/closed each finger is relative to the palm),
    which the raw per-point coordinates only encode indirectly.
    Must be applied identically at inference time.
    """
    raw_features = np.asarray(raw_features, dtype=np.float32)

    coordinates = raw_features[:, :63].reshape(-1, 21, 3)

    pairwise = []
    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):
            distance = np.linalg.norm(
                coordinates[:, FINGERTIPS[i]] - coordinates[:, FINGERTIPS[j]],
                axis=1
            )
            pairwise.append(distance)
    tip_to_tip = np.stack(pairwise, axis=1)

    tip_to_wrist = np.linalg.norm(
        coordinates[:, FINGERTIPS] - coordinates[:, 0:1],
        axis=2
    )

    tip_to_knuckle = np.linalg.norm(
        coordinates[:, FINGERTIPS] - coordinates[:, KNUCKLES],
        axis=2
    )

    engineered = np.concatenate(
        [tip_to_tip, tip_to_wrist, tip_to_knuckle],
        axis=1
    )

    return np.concatenate([raw_features, engineered], axis=1)


def evaluate_model(name, model, x_train, x_test, y_train, y_test, sample_weight=None):
    """Train one model and return its accuracy and predictions."""
    print(f"\nTraining {name}...")

    if sample_weight is not None:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"{name} accuracy: {accuracy:.4f}")

    return accuracy, predictions, model


def main():
    """Train multiple models and save the highest-accuracy classifier."""
    dataframe = pd.read_csv(DATA_PATH)

    feature_columns = [
        column
        for column in dataframe.columns
        if column.startswith("feature_")
    ]

    features = engineer_features(dataframe[feature_columns].values)
    labels = dataframe["label"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # Computed manually and passed via fit(sample_weight=...) instead of
    # class_weight="balanced" on the estimator — the latter routes
    # through sklearn's internal class-weight validation logic, which
    # has known edge-case bugs (see scikit-learn issues #22413, #29568).
    # Passing precomputed sample weights is equivalent but avoids that
    # code path entirely.
    tree_sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE
                    )
                )
            ]
        )
    }

    results = {}
    trained_models = {}
    predictions_by_model = {}

    for name, model in models.items():
        weight = tree_sample_weight if name in ("Random Forest", "Extra Trees") else None

        accuracy, predictions, trained_model = evaluate_model(
            name,
            model,
            x_train,
            x_test,
            y_train,
            y_test,
            sample_weight=weight
        )

        results[name] = accuracy
        trained_models[name] = trained_model
        predictions_by_model[name] = predictions

    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]
    best_predictions = predictions_by_model[best_model_name]

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": feature_columns,
            "model_name": best_model_name
        },
        MODEL_OUTPUT_PATH
    )

    report = classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )

    matrix = confusion_matrix(y_test, best_predictions)

    with RESULTS_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for name, accuracy in results.items():
            file.write(f"{name}: {accuracy:.4f}\n")

        file.write(f"\nBest model: {best_model_name}\n")
        file.write("\nClassification report:\n")
        file.write(report)
        file.write("\nConfusion matrix:\n")
        file.write(str(matrix))

    print(f"\nBest model: {best_model_name}")
    print(f"Best accuracy: {results[best_model_name]:.4f}")
    print(report)
    print(f"Saved model to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()