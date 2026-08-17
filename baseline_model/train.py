
import json
import torch
import torch.nn as nn
import torch.optim as optim
from baseline_model.dataloader import asl_data, train_loader, val_loader, test_loader
from baseline_model.model import ASLCNN
 
# Initialize model
num_classes = len(asl_data.classes)
model = ASLCNN(num_classes)
 
 
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
 
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
 
 
# Training
num_epochs = 10
 
for epoch in range(num_epochs):
 
    # Put model in training mode
    model.train()
 
    running_loss = 0.0
 
    # Loop through training batches
    for images, labels in train_loader:
 
        # Forward pass
        outputs = model(images)
 
        # Compute loss
        loss = criterion(outputs, labels)
 
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
 
        # Update weights
        optimizer.step()
 
        running_loss += loss.item()
 
    # Average training loss
    avg_loss = running_loss / len(train_loader)
 
    # Validation
    model.eval()
 
    correct = 0
    total = 0
 
    with torch.no_grad():
 
        for images, labels in val_loader:
 
            outputs = model(images)
 
            _, predicted = torch.max(outputs, dim=1)
 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
    val_accuracy = 100 * correct / total
 
    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Loss: {avg_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )
 
# Final Test Evaluation
model.eval()
 
correct = 0
total = 0
 
with torch.no_grad():
 
    for images, labels in test_loader:
 
        outputs = model(images)
 
        _, predicted = torch.max(outputs, dim=1)
 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
test_accuracy = 100 * correct / total
 
print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
 
# Save model
torch.save(model.state_dict(), "asl_cnn_model.pth")
 
# Save the class list (index -> letter) so inference scripts (like
# mediapipeModel.py) know how to map model outputs back to letters,
# and in the exact order the model was trained on (J/Z excluded).
with open("asl_classes.json", "w") as f:
    json.dump(asl_data.classes, f)
 
print("Model saved as asl_cnn_model.pth")
print("Class list saved as asl_classes.json")
 
