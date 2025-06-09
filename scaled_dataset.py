import torch
from torch.utils.data import Dataset

class ScaledDataset(Dataset):
    def __init__(self, X, y, scaler):
        self.X = X
        self.y = y
        self.scaler = scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x_scaled = self.scaler.transform([x])[0]  # avoid reshape/squeeze
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return torch.tensor(x_scaled, dtype=torch.float32), y