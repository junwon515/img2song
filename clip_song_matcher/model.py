import torch.nn as nn
from clip_song_matcher.config import INPUT_DIM, PROJ_DIM


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = INPUT_DIM, out_dim: int = PROJ_DIM):
        super().__init__()
        hidden_dim = 2048
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)