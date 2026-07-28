from pathlib import Path
import copy
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

# Change this path if your class folders are somewhere else.
DATA_DIR = Path("data/raw")

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

MODEL_OUTPUT_PATH = Path("models/aaliya_baseline.pth")


# --------------------------------------------------
# 2. REPRODUCIBILITY
# --------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------
# 3. MODEL
# --------------------------------------------------

class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------
# 4. TRAINING / VALIDATION FUNCTIONS
# --------------------------------------------------

def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(images)
        loss = loss_function(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (
            predictions.argmax(dim=1) == labels
        ).sum().item()
        total_examples += images.size(0)

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


@torch.no_grad()
def run_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = loss_function(predictions, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (
            predictions.argmax(dim=1) == labels
        ).sum().item()
        total_examples += images.size(0)

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


# --------------------------------------------------
# 5. MAIN PROGRAM
# --------------------------------------------------

def main() -> None:
    set_seed(RANDOM_SEED)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"\nDataset folder not found: {DATA_DIR}\n"
            "Change DATA_DIR near the top of train_baseline.py "
            "so it points to the folder containing the class folders."
        )

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Dataset path: {DATA_DIR.resolve()}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )

    # Load the same files twice because training and validation
    # need different transforms.
    training_source = datasets.ImageFolder(
        root=DATA_DIR,
        transform=train_transform,
    )

    validation_source = datasets.ImageFolder(
        root=DATA_DIR,
        transform=validation_transform,
    )

    number_of_examples = len(training_source)
    number_of_classes = len(training_source.classes)

    if number_of_examples == 0:
        raise RuntimeError("The dataset contains no recognized images.")

    print(f"Classes: {training_source.classes}")
    print(f"Number of classes: {number_of_classes}")
    print(f"Total images: {number_of_examples}")

    indices = torch.randperm(
        number_of_examples,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    ).tolist()

    validation_size = int(
        number_of_examples * VALIDATION_FRACTION
    )
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    training_dataset = Subset(
        training_source,
        training_indices,
    )

    validation_dataset = Subset(
        validation_source,
        validation_indices,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training images: {len(training_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")

    model = BaselineCNN(
        num_classes=number_of_classes
    ).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_accuracy = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        training_loss, training_accuracy = run_training_epoch(
            model=model,
            loader=training_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = run_validation_epoch(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        training_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss: {training_loss:.4f} | "
            f"train accuracy: {training_accuracy * 100:.2f}% | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy * 100:.2f}%"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_state = copy.deepcopy(
                model.state_dict()
            )

    elapsed_seconds = time.time() - start_time

    MODEL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": best_model_state,
            "class_names": training_source.classes,
            "image_size": IMAGE_SIZE,
            "best_validation_accuracy": best_validation_accuracy,
        },
        MODEL_OUTPUT_PATH,
    )

    print()
    print("Training finished.")
    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )
    print(f"Training time: {elapsed_seconds / 60:.2f} minutes")
    print(f"Saved model: {MODEL_OUTPUT_PATH}")

    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure()
    plt.plot(epochs, training_losses, label="Training loss")
    plt.plot(epochs, validation_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline CNN Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, training_accuracies, label="Training accuracy")
    plt.plot(
        epochs,
        validation_accuracies,
        label="Validation accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline CNN Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_accuracy.png")
    plt.close()

    print("Saved plots:")
    print("  baseline_loss.png")
    print("  baseline_accuracy.png")


if __name__ == "__main__":
    main()

