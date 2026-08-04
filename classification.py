import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = "/Users/aashutosh/Documents/Kaggle/asl_dataset/asl_dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 60
NUM_CLASSES = 24

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
print(train_ds.class_names)
print(len(train_ds.class_names))

labels = []

for _, y in train_ds.unbatch():
    labels.append(int(y.numpy()))

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weight_dict = {
    i: class_weights[i]
    for i in range(len(class_weights))
}

print(class_weight_dict)

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

train_ds = (
    train_ds
    .cache()
    .shuffle(1000)
    .prefetch(AUTOTUNE)
)

validation_ds = (
    validation_ds
    .cache()
    .prefetch(AUTOTUNE)
)

# ============================================================
# Data Augmentation
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.05),
])


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

    inputs = layers.Input(shape=IMG_SIZE + (3,))

    x = data_augmentation(inputs)

    x = base_model(
        x,
        training=False
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        256,
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
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=3e-4,
            weight_decay=1e-5,
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model, base_model


# ============================================================
# Create Model
# ============================================================

model, base_model = build_model(NUM_CLASSES)

print("\nModel Summary:")
model.summary()


# ============================================================
# Callbacks
# ============================================================

callbacks = [

    # Stop training if validation accuracy stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
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
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
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
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# ============================================================
# Print Accuracy History
# ============================================================

# Combine histories from initial training + fine-tuning
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

print("\nTraining Accuracy History:")
print(train_acc)

print("\nValidation Accuracy History:")
print(val_acc)


# ============================================================
# Plot Training Curves
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_acc,
    label="Training Accuracy"
)

plt.plot(
    val_acc,
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
# Generate Predictions
# ============================================================

y_true = []
y_pred = []

for images, labels in validation_ds:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


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