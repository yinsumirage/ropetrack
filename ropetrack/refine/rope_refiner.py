from __future__ import annotations

import torch
from torch import nn


class RopePoseRefiner(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(60, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 45),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        base_hand_pose: torch.Tensor,
        base_rope_norm: torch.Tensor,
        input_rope_norm: torch.Tensor,
        rope_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([base_hand_pose, base_rope_norm, input_rope_norm, rope_valid], dim=1)
        delta_hand_pose = self.net(x)
        return base_hand_pose + delta_hand_pose, delta_hand_pose
