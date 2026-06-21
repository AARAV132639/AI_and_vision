import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.mlp1 = nn.Linear(3,64)

        self.mlp2 = nn.Linear(64,128)

        self.mlp3 = nn.Linear(128,1024)

    def forward(self,x):

        # x [B,N,3]

        x = F.relu(
            self.mlp1(x)
        )

        x = F.relu(
            self.mlp2(x)
        )

        x = self.mlp3(x)

        z = torch.max(
            x,
            dim=1
        )[0]

        return z
    
