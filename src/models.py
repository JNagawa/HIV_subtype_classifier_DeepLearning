import torch
import torch.nn as nn


class HIV_MLP(nn.Module):
    """
    Baseline MLP: Evaluates global nucleotide composition across the entire sequence.
    """

    def __init__(self, num_classes, dropout=0.3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # Global Average Pooling: squashes the entire sequence into 1 average value per channel
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


class HIV_CNN(nn.Module):
    """1D-CNN for HIV pol gene subtype classification."""

    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(1),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(1),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(1),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class HIV_BiLSTM(nn.Module):
    """Bidirectional LSTM for HIV pol gene subtype classification."""

    def __init__(
        self, num_classes, vocab=5, embed=128, hidden=256, layers=2, dropout=0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed, padding_idx=5)
        self.lstm = nn.LSTM(
            embed,
            hidden,
            layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, (h, _) = self.lstm(emb)
        out = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(out))


def get_model(model_name, num_classes=4, dropout=0.3, device="cpu"):
    models = {
        "mlp": HIV_MLP(num_classes, dropout=dropout),
        "cnn": HIV_CNN(num_classes, dropout=dropout),
        "bilstm": HIV_BiLSTM(num_classes, vocab=6, dropout=dropout),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name].to(device)
