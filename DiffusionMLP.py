# diffusion_mlp.py
import torch
import torch.nn as nn
from SinusoidalPosEmb import SinusoidalPosEmb

class DiffusionMLP(nn.Module):
    def __init__(self, state_dim, action_dim, time_dim=16, hidden_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)

        )

    def forward(self, state, noisy_action, time_step):
        time_emb = self.time_mlp(time_step)
        x = torch.cat([state, noisy_action, time_emb], dim=-1)
        return self.net(x)
