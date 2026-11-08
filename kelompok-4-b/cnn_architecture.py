import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class AudioCNN1D(nn.Module):
    """Deep Learning 1D-CNN untuk deteksi suara manusia vs AI berbasis 20 fitur akustik."""
    def __init__(self, in_features=20, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (in_features // 4), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNVoiceClassifier:
    """Wrapper scikit-learn compatible untuk inferensi AudioCNN1D di Flask & Notebook."""
    def __init__(self, in_features=20, num_classes=2, lr=0.003, epochs=50):
        self.in_features = in_features
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.model = AudioCNN1D(in_features, num_classes)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

