import torch
from torch.utils.data import Dataset


class IndexedScaledDataset(Dataset):
    def __init__(self, X, y, indices, scaler=None):
        self.X = X
        self.y = y
        self.indices = indices
        self.scaler = scaler

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.X[i]
        if self.scaler:
            x = self.scaler.transform([x])[0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[i], dtype=torch.float32).unsqueeze(0)