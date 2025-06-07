import torch.nn as nn
import torch.optim as optim

class Model_v1_2018(nn.Module):
    def __init__(self, input_dim=2381):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024), 
            nn.Dropout(0.5), 
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),  
            #nn.Linear(128, 64),
            #nn.ReLU(),
            #nn.BatchNorm1d(64),
            

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)