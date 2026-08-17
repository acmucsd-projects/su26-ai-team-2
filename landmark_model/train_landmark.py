"""
Trains the landmark-based MLP on features extracted by extract_landmarks.py.

Run from the project root:
    python -m landmark_model.train_landmark
"""
import csv
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from landmark_model.landmark_model import ASLLandmarkMLP

CSV_PATH = "landmarks.csv"


class LandmarkDataset(Dataset):
    def __init__(self, csv_path):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # header
            rows = list(reader)

        self.classes = sorted(set(row[0] for row in rows))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        features = [[float(x) for x in row[1:]] for row in rows]
        labels = [self.class_to_idx[row[0]] for row in rows]

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def main():
    dataset = LandmarkDataset(CSV_PATH)
    num_classes = len(dataset.classes)
    print(f"Loaded {len(dataset)} samples across {num_classes} classes")

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    model = ASLLandmarkMLP(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for feats, labels in train_loader:
            outputs = model(feats)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                outputs = model(feats)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

    # Final test evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for feats, labels in test_loader:
            outputs = model(feats)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

    torch.save(model.state_dict(), "asl_landmark_model.pth")
    with open("asl_landmark_classes.json", "w") as f:
        json.dump(dataset.classes, f)

    print("Model saved as asl_landmark_model.pth")
    print("Classes saved as asl_landmark_classes.json")


if __name__ == "__main__":
    main()