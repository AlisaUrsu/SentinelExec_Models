import torch
import torch.nn as nn
import torch.optim as optim

class Model_BIG_v4(nn.Module):
    def __init__(self, input_dim=2351):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.net(x)