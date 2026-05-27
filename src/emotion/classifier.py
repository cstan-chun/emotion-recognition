import torch
import torch.nn as nn


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128,
                 num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
