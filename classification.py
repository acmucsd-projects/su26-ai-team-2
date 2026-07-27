import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = "/Users/aashutosh/Documents/Kaggle/asl_dataset/asl_dataset"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 36

MODEL_PATH = "asl_baseline_model.keras"


# ============================================================
# Load Dataset
# ============================================================

print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.30,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.30,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

# Class names determined by the directory structure
class_names = train_ds.class_names

print("\nClasses:")
print(class_names)

print("\nNumber of classes:", len(class_names))


# ============================================================
# Performance optimization
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)


# ============================================================
# Data Augmentation
# ============================================================

data_augmentation = models.Sequential([
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.05),
], name="augmentation")


# ============================================================
# Build EfficientNetB0 Model
# ============================================================

def build_model(num_classes):

    # Pretrained ImageNet feature extractor
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,)
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    inputs = layers.Input(
        shape=IMG_SIZE + (3,)
    )

    # Data augmentation
    x = data_augmentation(inputs)

    # IMPORTANT:
    # Do NOT add:
    #
    # layers.Rescaling(1.0 / 255)
    #
    # EfficientNetB0 already handles its own input rescaling.

    x = base_model(
        x,
        training=False
    )

    # Convert feature maps into a single feature vector
    x = layers.GlobalAveragePooling2D()(x)

    # Reduce overfitting
    x = layers.Dropout(0.3)(x)

    # Classification layer
    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="asl_efficientnet_b0"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# Create Model
# ============================================================

model = build_model(NUM_CLASSES)

print("\nModel Summary:")
model.summary()


# ============================================================
# Callbacks
# ============================================================

callbacks = [

    # Stop training if validation accuracy stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    ),

    # Save the best model during training
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True
    ),

    # Reduce learning rate if validation accuracy plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )
]


# ============================================================
# Train Model
# ============================================================

print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# Print Accuracy History
# ============================================================

print("\nTraining Accuracy History:")
print(history.history["accuracy"])

print("\nValidation Accuracy History:")
print(history.history["val_accuracy"])


# ============================================================
# Plot Training Curves
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "training_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved training_curves.png")


# ============================================================
# Evaluate on Validation Set
# ============================================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    validation_ds,
    verbose=1
)

print(
    f"\nTest accuracy: {test_accuracy:.4f}"
    f"   |   Test loss: {test_loss:.4f}"
)


# ============================================================
# Generate Predictions
# ============================================================

y_true = []
y_pred = []

for images, labels in validation_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# Classification Report
# ============================================================

print("\nClassification report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(14, 12)
)

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=90
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved confusion_matrix.png")


# ============================================================
# Save Final Model
# ============================================================

model.save(
    MODEL_PATH
)

print(
    f"Saved model to {MODEL_PATH}"
)