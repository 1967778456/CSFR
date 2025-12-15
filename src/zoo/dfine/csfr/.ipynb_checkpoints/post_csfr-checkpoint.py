import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .utils import local_entropy

__all__ = ["PostCSFR"]


class CrossScaleTokenAlign(nn.Module):
    """
    跨尺度 Token 一致性校准（无 einops 版本）
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        feats: List of [B, C, H, W]
        """
        tokens = []
        shapes = []

        for f in feats:
            B, C, H, W = f.shape
            tokens.append(f.flatten(2).transpose(1, 2))  # [B, HW, C]
            shapes.append((H, W))

        x = torch.cat(tokens, dim=1)  # [B, sum(HW), C]

        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        # Split back to multi-scale features
        outs = []
        idx = 0
        for (H, W) in shapes:
            n = H * W
            feat = x[:, idx:idx + n, :]           # [B, HW, C]
            feat = feat.transpose(1, 2).reshape(-1, self.channels, H, W)
            outs.append(feat)
            idx += n

        return outs


class TokenStructureRefine(nn.Module):
    """
    单尺度 Token 结构精炼
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GroupNorm(max(1, channels // 4), channels),
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        entropy = local_entropy(x)
        gate = self.gate(entropy)
        return x + self.conv(x) * gate


class PostCSFR(nn.Module):
    """
    编码器输出阶段的一致性优化模块
    """
    def __init__(self, channels_list: List[int], num_heads: int = 4):
        super().__init__()

        self.align = CrossScaleTokenAlign(
            channels=channels_list[-1],
            num_heads=num_heads
        )

        self.refine_blocks = nn.ModuleList([
            TokenStructureRefine(c) for c in channels_list
        ])

        self.norms = nn.ModuleList([
            nn.GroupNorm(max(1, c // 4), c) for c in channels_list
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        feats = self.align(feats)

        outs = []
        for f, refine, norm in zip(feats, self.refine_blocks, self.norms):
            f = refine(f)
            f = norm(f)
            outs.append(f)

        return outs
