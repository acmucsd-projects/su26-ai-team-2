import torch.nn as nn


class ASLLandmarkMLP(nn.Module):
    """Classifies a 63-dim normalized hand-landmark vector (21 points x,y,z)."""
    def __init__(self, num_classes, input_dim=63):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)