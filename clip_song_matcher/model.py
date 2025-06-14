import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 512):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)
