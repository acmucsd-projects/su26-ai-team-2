# su26-ai-team-2

Baseline CNN

Dataset:
- ASL Kaggle Dataset
- 36 classes (A-Z, 0-9)

Preprocessing:
- Resize: 128x128
- Normalize: mean=0.5, std=0.5

Model:
- Custom CNN
- Conv2d layers
- Max pooling
- Fully connected classifier

Training:
- Epochs: 10
- Batch size: 32
- Optimizer: Adam
- Learning rate: 0.001
- Loss: CrossEntropyLoss

Test Accuracy:
- 95.46%