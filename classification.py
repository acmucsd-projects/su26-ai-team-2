import os
import json
from pathlib import Path
from collections import Counter, deque

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

from tensorflow.keras import layers, models

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "/Users/aashutosh/Documents/Kaggle/asl_dataset/asl_dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

VALIDATION_SPLIT = 0.30

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 50

# EfficientNetB0 layer from which fine-tuning begins.
FINE_TUNE_FROM = 150

MODEL_PATH = "asl_baseline_model.keras"

CLASS_NAMES_PATH = "class_names.json"

INFERENCE_CONFIG_PATH = "inference_config.json"

SEED = 42


# ============================================================
# REAL-TIME INFERENCE CONFIGURATION
# ============================================================

INFERENCE_CONFIG = {

    "window_size": 15,

    "min_stable_predictions": 8,

    "min_confidence": 0.45,

    "min_margin": 0.15,

    "min_consensus_ratio": 0.60,

    "suppress_duplicate": True,

    "recency_weight": 1.15,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# TEMPORAL PREDICTION DECODER
# ============================================================

class TemporalPredictionDecoder:

    def __init__(
        self,
        class_names,
        window_size=None,
        min_stable_predictions=None,
        min_confidence=None,
        min_margin=None,
        min_consensus_ratio=None,
        suppress_duplicate=None,
        recency_weight=None,
    ):

        self.class_names = list(class_names)

        self.window_size = (
            window_size
            if window_size is not None
            else INFERENCE_CONFIG["window_size"]
        )

        self.min_stable_predictions = (
            min_stable_predictions
            if min_stable_predictions is not None
            else INFERENCE_CONFIG["min_stable_predictions"]
        )

        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else INFERENCE_CONFIG["min_confidence"]
        )

        self.min_margin = (
            min_margin
            if min_margin is not None
            else INFERENCE_CONFIG["min_margin"]
        )

        self.min_consensus_ratio = (
            min_consensus_ratio
            if min_consensus_ratio is not None
            else INFERENCE_CONFIG["min_consensus_ratio"]
        )

        self.suppress_duplicate = (
            suppress_duplicate
            if suppress_duplicate is not None
            else INFERENCE_CONFIG["suppress_duplicate"]
        )

        self.recency_weight = (
            recency_weight
            if recency_weight is not None
            else INFERENCE_CONFIG["recency_weight"]
        )

        self.probability_history = deque(
            maxlen=self.window_size
        )

        self.prediction_history = deque(
            maxlen=self.window_size
        )

        self.last_accepted_label = None

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.probability_history.clear()

        self.prediction_history.clear()

        self.last_accepted_label = None

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(self, probabilities):

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        if probabilities.ndim != 1:

            raise ValueError(
                "Expected a 1D probability vector."
            )

        if len(probabilities) != len(self.class_names):

            raise ValueError(
                "Probability vector length does not "
                "match number of classes."
            )

        probability_sum = np.sum(probabilities)

        if probability_sum <= 0:

            return {
                "accepted": False,
                "uncertain": True,
                "label": None,
                "confidence": 0.0,
                "margin": 0.0,
            }

        probabilities = (
            probabilities / probability_sum
        )

        # ----------------------------------------------------
        # RAW PREDICTION
        # ----------------------------------------------------

        raw_index = int(
            np.argmax(probabilities)
        )

        raw_confidence = float(
            probabilities[raw_index]
        )

        sorted_indices = np.argsort(
            probabilities
        )[::-1]

        top1_index = int(
            sorted_indices[0]
        )

        top2_index = int(
            sorted_indices[1]
        )

        top1_probability = float(
            probabilities[top1_index]
        )

        top2_probability = float(
            probabilities[top2_index]
        )

        raw_margin = (
            top1_probability
            - top2_probability
        )

        # ----------------------------------------------------
        # STORE PREDICTION
        # ----------------------------------------------------

        self.probability_history.append(
            probabilities.copy()
        )

        self.prediction_history.append(
            raw_index
        )

        # ----------------------------------------------------
        # NOT ENOUGH HISTORY
        # ----------------------------------------------------

        if (
            len(self.probability_history)
            < self.min_stable_predictions
        ):

            return {
                "accepted": False,
                "uncertain": True,
                "label": None,
                "raw_label": self.class_names[raw_index],
                "confidence": raw_confidence,
                "margin": raw_margin,
                "consensus": 0.0,
            }

        # ----------------------------------------------------
        # RECENCY WEIGHTING
        # ----------------------------------------------------

        history = np.asarray(
            self.probability_history
        )

        n = len(history)

        weights = np.array(
            [
                self.recency_weight ** i
                for i in range(n)
            ],
            dtype=np.float32,
        )

        weights /= np.sum(weights)

        weighted_probabilities = np.sum(
            history * weights[:, None],
            axis=0,
        )

        # ----------------------------------------------------
        # TEMPORAL TOP-1 / TOP-2
        # ----------------------------------------------------

        temporal_sorted = np.argsort(
            weighted_probabilities
        )[::-1]

        temporal_top1 = int(
            temporal_sorted[0]
        )

        temporal_top2 = int(
            temporal_sorted[1]
        )

        temporal_confidence = float(
            weighted_probabilities[
                temporal_top1
            ]
        )

        temporal_second_confidence = float(
            weighted_probabilities[
                temporal_top2
            ]
        )

        temporal_margin = (
            temporal_confidence
            - temporal_second_confidence
        )

        temporal_label = self.class_names[
            temporal_top1
        ]

        # ----------------------------------------------------
        # MAJORITY CONSENSUS
        # ----------------------------------------------------

        prediction_counts = Counter(
            self.prediction_history
        )

        majority_index, majority_count = (
            prediction_counts.most_common(1)[0]
        )

        consensus_ratio = (
            majority_count
            / len(self.prediction_history)
        )

        # ----------------------------------------------------
        # ACCEPTANCE CONDITIONS
        # ----------------------------------------------------

        enough_stability = (
            majority_count
            >= self.min_stable_predictions
        )

        enough_consensus = (
            consensus_ratio
            >= self.min_consensus_ratio
        )

        enough_confidence = (
            temporal_confidence
            >= self.min_confidence
        )

        enough_margin = (
            temporal_margin
            >= self.min_margin
        )

        predictions_agree = (
            majority_index
            == temporal_top1
        )

        accepted = (
            enough_stability
            and enough_consensus
            and enough_confidence
            and enough_margin
            and predictions_agree
        )

        # ----------------------------------------------------
        # DUPLICATE SUPPRESSION
        # ----------------------------------------------------

        if (
            self.suppress_duplicate
            and accepted
            and self.last_accepted_label
            == temporal_label
        ):

            accepted = False

        # ----------------------------------------------------
        # ACCEPTED
        # ----------------------------------------------------

        if accepted:

            self.last_accepted_label = (
                temporal_label
            )

            self.probability_history.clear()

            self.prediction_history.clear()

            return {
                "accepted": True,
                "uncertain": False,
                "label": temporal_label,
                "raw_label": self.class_names[
                    raw_index
                ],
                "confidence": temporal_confidence,
                "margin": temporal_margin,
                "consensus": consensus_ratio,
            }

        # ----------------------------------------------------
        # UNCERTAIN
        # ----------------------------------------------------

        return {
            "accepted": False,
            "uncertain": True,
            "label": None,
            "raw_label": self.class_names[
                raw_index
            ],
            "confidence": temporal_confidence,
            "margin": temporal_margin,
            "consensus": consensus_ratio,
        }


# ============================================================
# LOAD INFERENCE DECODER
# ============================================================

def load_inference_decoder():

    if not os.path.exists(
        CLASS_NAMES_PATH
    ):

        raise FileNotFoundError(
            f"Could not find {CLASS_NAMES_PATH}"
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
    ) as f:

        class_names = json.load(f)

    return TemporalPredictionDecoder(
        class_names
    )


# ============================================================
# SAVE INFERENCE CONFIGURATION
# ============================================================

with open(
    INFERENCE_CONFIG_PATH,
    "w",
) as f:

    json.dump(
        INFERENCE_CONFIG,
        f,
        indent=4,
    )

print(
    f"Saved inference configuration to "
    f"{INFERENCE_CONFIG_PATH}"
)


# ============================================================
# LOAD FILES FROM DATASET
#
# We are deliberately NOT using Keras' automatic
# validation_split here.
#
# The previous evaluation produced:
#
# a-q -> support 0
# r-y -> support 502
#
# That made the 98% validation accuracy misleading.
#
# We therefore explicitly create a STRATIFIED split.
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

dataset_path = Path(
    DATASET_DIR
)

if not dataset_path.exists():

    raise FileNotFoundError(
        f"Dataset directory does not exist:\n"
        f"{DATASET_DIR}"
    )


# ------------------------------------------------------------
# CLASS NAMES
# ------------------------------------------------------------

class_directories = sorted(
    [
        directory
        for directory in dataset_path.iterdir()
        if directory.is_dir()
    ],
    key=lambda x: x.name.lower(),
)


if len(class_directories) == 0:

    raise ValueError(
        "No class directories were found in "
        f"{DATASET_DIR}"
    )


class_names = [
    directory.name
    for directory in class_directories
]

NUM_CLASSES = len(
    class_names
)


print("\nClasses:")

for i, name in enumerate(
    class_names
):

    print(
        f"{i}: {name}"
    )


print(
    f"\nNumber of classes: "
    f"{NUM_CLASSES}"
)


# ------------------------------------------------------------
# COLLECT IMAGE FILES
# ------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
}


all_file_paths = []

all_labels = []


for class_index, class_directory in enumerate(
    class_directories
):

    class_files = sorted(
        [
            path
            for path in class_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    print(
        f"{class_names[class_index]:>5}: "
        f"{len(class_files)} images"
    )

    for path in class_files:

        all_file_paths.append(
            str(path)
        )

        all_labels.append(
            class_index
        )


all_file_paths = np.asarray(
    all_file_paths
)

all_labels = np.asarray(
    all_labels,
    dtype=np.int32,
)


print(
    f"\nTotal images: "
    f"{len(all_file_paths)}"
)


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

train_paths, validation_paths, train_labels, validation_labels = (
    train_test_split(
        all_file_paths,
        all_labels,
        test_size=VALIDATION_SPLIT,
        random_state=SEED,
        stratify=all_labels,
        shuffle=True,
    )
)


print(
    f"\nTraining images: "
    f"{len(train_paths)}"
)

print(
    f"Validation images: "
    f"{len(validation_paths)}"
)


# ============================================================
# VERIFY THAT EVERY CLASS EXISTS IN BOTH SPLITS
# ============================================================

train_class_counts = Counter(
    train_labels
)

validation_class_counts = Counter(
    validation_labels
)


print(
    "\nTraining class distribution:"
)

for i, class_name in enumerate(
    class_names
):

    print(
        f"{class_name:>5}: "
        f"{train_class_counts[i]}"
    )


print(
    "\nValidation class distribution:"
)

for i, class_name in enumerate(
    class_names
):

    print(
        f"{class_name:>5}: "
        f"{validation_class_counts[i]}"
    )


missing_train_classes = [
    class_names[i]
    for i in range(NUM_CLASSES)
    if train_class_counts[i] == 0
]


missing_validation_classes = [
    class_names[i]
    for i in range(NUM_CLASSES)
    if validation_class_counts[i] == 0
]


if missing_train_classes:

    raise ValueError(
        "The following classes are missing from "
        "the training split:\n"
        + ", ".join(missing_train_classes)
    )


if missing_validation_classes:

    raise ValueError(
        "The following classes are missing from "
        "the validation split:\n"
        + ", ".join(missing_validation_classes)
    )


print(
    "\nAll classes are present in both "
    "training and validation."
)


# ============================================================
# SAVE EXACT CLASS ORDER
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
) as f:

    json.dump(
        class_names,
        f,
        indent=4,
    )


print(
    f"\nSaved {CLASS_NAMES_PATH}"
)


# ============================================================
# IMAGE LOADING FUNCTION
# ============================================================

def load_image(
    file_path,
    label,
):

    image = tf.io.read_file(
        file_path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape(
        [
            None,
            None,
            3,
        ]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    return image, label


# ============================================================
# CREATE TF.DATA DATASETS
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels,
    )
)

validation_ds = tf.data.Dataset.from_tensor_slices(
    (
        validation_paths,
        validation_labels,
    )
)


train_ds = train_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)

validation_ds = validation_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)


train_ds = (
    train_ds
    .cache()
    .shuffle(
        1000,
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    .batch(BATCH_SIZE)
    .prefetch(
        tf.data.AUTOTUNE
    )
)


validation_ds = (
    validation_ds
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(
        tf.data.AUTOTUNE
    )
)


# ============================================================
# COMPUTE CLASS WEIGHTS
# ============================================================

print(
    "\nComputing class weights..."
)


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(
        NUM_CLASSES
    ),
    y=train_labels,
)


class_weight_dict = {
    i: float(
        class_weights[i]
    )
    for i in range(NUM_CLASSES)
}


print(
    "\nClass weights:"
)


for i, weight in class_weight_dict.items():

    print(
        f"{class_names[i]}: "
        f"{weight:.4f}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [

        layers.RandomRotation(
            0.05,
            fill_mode="reflect",
        ),

        layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08,
            fill_mode="reflect",
        ),

        layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10),
            fill_mode="reflect",
        ),

        layers.RandomContrast(
            0.15
        ),

        layers.RandomBrightness(
            0.10
        ),

    ],
    name="data_augmentation",
)


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    num_classes
):

    base_model = (
        tf.keras.applications.EfficientNetB0(

            include_top=False,

            weights="imagenet",

            input_shape=IMG_SIZE + (3,),
        )
    )


    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    base_model.trainable = False


    inputs = layers.Input(
        shape=IMG_SIZE + (3,),
        name="input_image",
    )


    x = data_augmentation(
        inputs
    )


    # EfficientNetB0 in tf.keras includes preprocessing
    # internally.
    #
    # Therefore we keep images in [0,255].
    #
    # DO NOT divide by 255 here.

    x = base_model(
        x,
        training=False,
    )


    x = layers.GlobalAveragePooling2D()(
        x
    )


    x = layers.Dropout(
        0.40
    )(x)


    x = layers.Dense(
        256,
        activation="relu",
    )(x)


    x = layers.Dropout(
        0.25
    )(x)


    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classification",
    )(x)


    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="asl_efficientnet_b0",
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT use:
    #
    # SparseCategoricalCrossentropy(
    #     label_smoothing=0.05
    # )
    #
    # Your installed TensorFlow version does not support
    # label_smoothing for SparseCategoricalCrossentropy.
    #
    # We therefore use standard sparse categorical CE.
    # --------------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=3e-4,
            weight_decay=1e-5,
        ),

        loss=tf.keras.losses.SparseCategoricalCrossentropy(),

        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )


    return model, base_model


# ============================================================
# CREATE MODEL
# ============================================================

model, base_model = build_model(
    NUM_CLASSES
)


print(
    "\n" + "=" * 60
)

print(
    "MODEL SUMMARY"
)

print(
    "=" * 60
)


model.summary()


# ============================================================
# STAGE 1 CALLBACKS
# ============================================================

callbacks_stage1 = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=6,

        mode="max",

        restore_best_weights=True,

        verbose=1,
    ),

    tf.keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=2,

        min_lr=1e-7,

        verbose=1,
    ),
]


# ============================================================
# STAGE 1
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "STAGE 1: TRAINING CLASSIFICATION HEAD"
)

print(
    "=" * 60
)


history_stage1 = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=INITIAL_EPOCHS,

    callbacks=callbacks_stage1,

    class_weight=class_weight_dict,
)


# ============================================================
# STAGE 2
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "STAGE 2: FINE-TUNING EFFICIENTNET"
)

print(
    "=" * 60
)


base_model.trainable = True


print(
    f"Fine-tuning from EfficientNet "
    f"layer {FINE_TUNE_FROM} onward."
)


# ------------------------------------------------------------
# FREEZE EARLY LAYERS
# ------------------------------------------------------------

for layer in base_model.layers[
    :FINE_TUNE_FROM
]:

    layer.trainable = False


# ------------------------------------------------------------
# KEEP BATCH NORMALIZATION FROZEN
# ------------------------------------------------------------

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization,
    ):

        layer.trainable = False


# ------------------------------------------------------------
# RECOMPILE
# ------------------------------------------------------------

model.compile(

    optimizer=tf.keras.optimizers.AdamW(

        learning_rate=1e-5,

        weight_decay=1e-5,
    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[

        tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ],
)


# ============================================================
# STAGE 2 CALLBACKS
# ============================================================

callbacks_stage2 = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=8,

        mode="max",

        restore_best_weights=True,

        verbose=1,
    ),

    tf.keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=3,

        min_lr=1e-7,

        verbose=1,
    ),
]


# ============================================================
# TRAIN STAGE 2
# ============================================================

history_stage2 = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=callbacks_stage2,

    class_weight=class_weight_dict,
)


# ============================================================
# COMBINE HISTORIES
# ============================================================

train_acc = (
    history_stage1.history["accuracy"]
    +
    history_stage2.history["accuracy"]
)

val_acc = (
    history_stage1.history["val_accuracy"]
    +
    history_stage2.history["val_accuracy"]
)

train_loss = (
    history_stage1.history["loss"]
    +
    history_stage2.history["loss"]
)

val_loss = (
    history_stage1.history["val_loss"]
    +
    history_stage2.history["val_loss"]
)


# ============================================================
# BEST VALIDATION ACCURACY
# ============================================================

best_epoch_index = int(
    np.argmax(val_acc)
)

best_val_accuracy = float(
    val_acc[best_epoch_index]
)


print(
    "\n" + "=" * 60
)

print(
    "TRAINING RESULTS"
)

print(
    "=" * 60
)


print(
    f"\nBest validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best epoch: "
    f"{best_epoch_index + 1}"
)


# ============================================================
# TRAINING ACCURACY CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    train_acc,
    label="Training Accuracy",
)


plt.plot(
    val_acc,
    label="Validation Accuracy",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.grid(True)


plt.savefig(
    "training_curves.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "\nSaved training_curves.png"
)


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    train_loss,
    label="Training Loss",
)


plt.plot(
    val_loss,
    label="Validation Loss",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.grid(True)


plt.savefig(
    "loss_curves.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "Saved loss_curves.png"
)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "LOADING BEST MODEL"
)

print(
    "=" * 60
)


if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(
        f"Could not find saved model: "
        f"{MODEL_PATH}"
    )


model = tf.keras.models.load_model(
    MODEL_PATH
)


print(
    f"Loaded best model from "
    f"{MODEL_PATH}"
)


# ============================================================
# VERIFY MODEL / CLASS COUNT
# ============================================================

model_outputs = (
    model.output_shape[-1]
)


if model_outputs != NUM_CLASSES:

    raise ValueError(

        "\nMODEL/CLASS MISMATCH\n"

        f"Model outputs: "
        f"{model_outputs}\n"

        f"Dataset classes: "
        f"{NUM_CLASSES}\n"
    )


# ============================================================
# GENERATE VALIDATION PREDICTIONS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "GENERATING VALIDATION PREDICTIONS"
)

print(
    "=" * 60
)


y_true = []

y_pred = []

y_probabilities = []


for images, labels_batch in validation_ds:

    predictions = model.predict(
        images,
        verbose=0,
    )


    predicted_classes = np.argmax(
        predictions,
        axis=1,
    )


    y_true.extend(
        labels_batch.numpy()
    )


    y_pred.extend(
        predicted_classes
    )


    y_probabilities.extend(
        predictions
    )


y_true = np.asarray(
    y_true,
    dtype=np.int32,
)

y_pred = np.asarray(
    y_pred,
    dtype=np.int32,
)

y_probabilities = np.asarray(
    y_probabilities
)


# ============================================================
# FINAL VALIDATION DISTRIBUTION CHECK
#
# This is specifically included to prevent the previous
# situation where the report showed:
#
# a-q = support 0
# r-y = support 502
#
# That must never happen with this stratified split.
# ============================================================

final_validation_counts = Counter(
    y_true
)


for class_index in range(
    NUM_CLASSES
):

    if (
        final_validation_counts[
            class_index
        ] == 0
    ):

        raise RuntimeError(
            "Validation evaluation error: "
            f"class '{class_names[class_index]}' "
            "has zero validation samples."
        )


print(
    "\nValidation contains all "
    f"{NUM_CLASSES} classes."
)


# ============================================================
# TOP-1 / TOP-3 ACCURACY
# ============================================================

top1_accuracy = float(
    np.mean(
        y_pred == y_true
    )
)


top3_accuracy = float(
    top_k_accuracy_score(

        y_true,

        y_probabilities,

        k=min(
            3,
            NUM_CLASSES,
        ),

        labels=np.arange(
            NUM_CLASSES
        ),
    )
)


print(
    "\n" + "=" * 60
)

print(
    "VALIDATION ACCURACY"
)

print(
    "=" * 60
)


print(
    f"Top-1 accuracy: "
    f"{top1_accuracy:.4f}"
)


print(
    f"Top-3 accuracy: "
    f"{top3_accuracy:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 60
)


all_labels = np.arange(
    NUM_CLASSES
)


report_dict = classification_report(

    y_true,

    y_pred,

    labels=all_labels,

    target_names=class_names,

    zero_division=0,

    output_dict=True,
)


report_text = classification_report(

    y_true,

    y_pred,

    labels=all_labels,

    target_names=class_names,

    zero_division=0,
)


print(
    report_text
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    "classification_report.txt",
    "w",
) as f:

    f.write(
        report_text
    )


print(
    "Saved classification_report.txt"
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "PER-CLASS ACCURACY"
)

print(
    "=" * 60
)


per_class_accuracy = {}


for i, class_name in enumerate(
    class_names
):

    class_mask = (
        y_true == i
    )


    total = int(
        np.sum(class_mask)
    )


    if total == 0:

        accuracy = 0.0

    else:

        correct = int(
            np.sum(
                y_pred[class_mask]
                == i
            )
        )

        accuracy = (
            correct / total
        )


    per_class_accuracy[
        class_name
    ] = float(
        accuracy
    )


    print(
        f"{class_name:>5}: "
        f"{accuracy:.4f}"
    )


# ============================================================
# SAVE PER-CLASS ACCURACY
# ============================================================

with open(
    "per_class_accuracy.json",
    "w",
) as f:

    json.dump(

        per_class_accuracy,

        f,

        indent=4,
    )


print(
    "\nSaved per_class_accuracy.json"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nGenerating confusion matrix..."
)


cm = confusion_matrix(

    y_true,

    y_pred,

    labels=all_labels,
)


plt.figure(
    figsize=(14, 12)
)


plt.imshow(
    cm
)


plt.title(
    "ASL Classification Confusion Matrix"
)


plt.colorbar()


plt.xticks(

    np.arange(
        NUM_CLASSES
    ),

    class_names,

    rotation=90,
)


plt.yticks(

    np.arange(
        NUM_CLASSES
    ),

    class_names,
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "True Label"
)


plt.tight_layout()


plt.savefig(

    "confusion_matrix.png",

    dpi=300,

    bbox_inches="tight",
)


plt.close()


print(
    "Saved confusion_matrix.png"
)


# ============================================================
# MOST COMMON CONFUSIONS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "MOST COMMON CLASS CONFUSIONS"
)

print(
    "=" * 60
)


confusions = []


for true_index in range(
    NUM_CLASSES
):

    for predicted_index in range(
        NUM_CLASSES
    ):

        if (
            true_index
            == predicted_index
        ):

            continue


        count = int(
            cm[
                true_index,
                predicted_index
            ]
        )


        if count > 0:

            confusions.append(

                (
                    count,

                    class_names[
                        true_index
                    ],

                    class_names[
                        predicted_index
                    ],
                )
            )


confusions.sort(
    reverse=True
)


print(
    "\nTop 20 confusion pairs:"
)


for (
    count,
    true_class,
    predicted_class,
) in confusions[:20]:

    print(
        f"{true_class} -> "
        f"{predicted_class}: "
        f"{count}"
    )


# ============================================================
# SAVE CONFUSION DATA
# ============================================================

confusion_data = [

    {
        "true_class":
            true_class,

        "predicted_class":
            predicted_class,

        "count":
            count,
    }

    for (
        count,
        true_class,
        predicted_class,
    )
    in confusions
]


with open(
    "confusion_pairs.json",
    "w",
) as f:

    json.dump(

        confusion_data,

        f,

        indent=4,
    )


print(
    "\nSaved confusion_pairs.json"
)


# ============================================================
# VERIFY TEMPORAL DECODER
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "VERIFYING TEMPORAL DECODER"
)

print(
    "=" * 60
)


decoder = TemporalPredictionDecoder(
    class_names
)


dummy_prediction = np.zeros(
    NUM_CLASSES,
    dtype=np.float32,
)


dummy_prediction[0] = 1.0


decoder_result = decoder.update(
    dummy_prediction
)


print(
    "Temporal decoder initialized successfully."
)


print(
    "Example result:"
)


print(
    decoder_result
)


# ============================================================
# FINAL MODEL INFORMATION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "FINAL MODEL"
)

print(
    "=" * 60
)


print(
    f"Model: "
    f"{MODEL_PATH}"
)


print(
    f"Input shape: "
    f"{model.input_shape}"
)


print(
    f"Output shape: "
    f"{model.output_shape}"
)


print(
    f"Number of classes: "
    f"{NUM_CLASSES}"
)


print(
    f"Top-1 validation accuracy: "
    f"{top1_accuracy:.4f}"
)


print(
    f"Top-3 validation accuracy: "
    f"{top3_accuracy:.4f}"
)


print(
    f"Best validation accuracy during training: "
    f"{best_val_accuracy:.4f}"
)


print(
    "\nGenerated files:"
)


print(
    f"  - {MODEL_PATH}"
)

print(
    f"  - {CLASS_NAMES_PATH}"
)

print(
    f"  - {INFERENCE_CONFIG_PATH}"
)

print(
    "  - training_curves.png"
)

print(
    "  - loss_curves.png"
)

print(
    "  - confusion_matrix.png"
)

print(
    "  - classification_report.txt"
)

print(
    "  - per_class_accuracy.json"
)

print(
    "  - confusion_pairs.json"
)


print(
    "\nTraining and evaluation complete."
)