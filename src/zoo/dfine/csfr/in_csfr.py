import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .utils import local_entropy, cross_scale_entropy_diff, EPS

__all__ = ["InCSFRBlock"]


# --------------------------------------------------
# FFT 频域特征（替代 torch_dct）
# --------------------------------------------------
def fft_energy(x: torch.Tensor) -> torch.Tensor:
    """
    计算频域能量图（幅值）
    x: [B, C, H, W]
    return: [B, 1, H, W]
    """
    fft = torch.fft.rfft2(x, norm="ortho")
    energy = torch.abs(fft)
    energy = energy.mean(dim=1, keepdim=True)
    return energy


# --------------------------------------------------
# 跨尺度注意力（语义引导）
# --------------------------------------------------
class CrossScaleAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(max(1, channels // 4), channels)

    def forward(self, low_feat, high_feat):
        B, C, H, W = low_feat.shape
        N = H * W

        q = self.q_proj(low_feat).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = self.k_proj(high_feat).view(B, self.num_heads, self.head_dim, N)
        v = self.v_proj(high_feat).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)

        out = self.norm(self.out_proj(out))
        return out + low_feat


# --------------------------------------------------
# 多因子动态融合（熵 + 频域）
# --------------------------------------------------
class MultiFactorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, low_feat, high_feat, ent_diff):
        freq_low = fft_energy(low_feat)
        freq_high = fft_energy(high_feat)

        freq_ratio = freq_low / (freq_high + EPS)
        weight = self.weight_net(torch.cat([ent_diff, freq_ratio], dim=1))

        return weight * high_feat + (1 - weight) * low_feat


import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import local_entropy, freq_energy_fft, EPS


class InCSFRBlock(nn.Module):
    """
    In-CSFR Block
    --------------
    跨尺度特征融合重构模块（FPN / PAN 通用）

    输入:
        low_feat : [B, C, H, W]   高分辨率
        high_feat: [B, C, h, w]   低分辨率
        ent_diff : [B, 1, H, W] or None

    输出:
        fused    : [B, C, H, W]
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels

        # 对齐投影
        self.low_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(channels // 4, channels),
            nn.GELU()
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(channels // 4, channels),
            nn.GELU()
        )

        # 融合权重预测（熵 + 频域）
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 融合后结构精炼
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GroupNorm(channels // 4, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(
        self,
        low_feat: torch.Tensor,
        high_feat: torch.Tensor,
        ent_diff: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        low_feat : [B,C,H,W]
        high_feat: [B,C,h,w]
        ent_diff : [B,1,H,W] or None
        """

        # 1. 上采样 high_feat → 对齐 low_feat
        high_up = F.interpolate(
            high_feat,
            size=low_feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # 2. 通道 / 语义对齐
        low = self.low_proj(low_feat)
        high = self.high_proj(high_up)

        # 3. 构建融合权重
        if ent_diff is None:
            ent_diff = torch.zeros(
                low.shape[0], 1, low.shape[2], low.shape[3],
                device=low.device, dtype=low.dtype
            )

        # 频域能量（FFT，稳定）
        freq_low = freq_energy_fft(low)
        freq_high = freq_energy_fft(high)
        freq_ratio = freq_low / (freq_high + EPS)

        # 融合权重 α ∈ [0,1]
        alpha = self.weight_predictor(
            torch.cat([ent_diff, freq_ratio], dim=1)
        )

        # 4. 动态融合
        fused = alpha * high + (1.0 - alpha) * low

        # 5. 结构重构 + 残差
        fused = fused + self.refine(fused)

        return fused
