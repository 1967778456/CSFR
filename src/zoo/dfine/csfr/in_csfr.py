import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import freq_energy_fft, EPS

__all__ = ["InCSFRBlock"]


class InCSFRBlock(nn.Module):
    """
    In-CSFR Block (master compatible)
    --------------------------------
    支持 ent_diff 的跨尺度特征重构模块
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        gn = max(1, channels // 4)

        self.low_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(gn, channels),
            nn.GELU()
        )

        self.high_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(gn, channels),
            nn.GELU()
        )

        self.weight_predictor = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(gn, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(
        self,
        low_feat: torch.Tensor,
        high_feat: torch.Tensor,
        ent_diff: torch.Tensor | None = None
    ):
        # 上采样 high → low
        high_up = F.interpolate(
            high_feat,
            size=low_feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        low = self.low_proj(low_feat)
        high = self.high_proj(high_up)

        if ent_diff is None:
            ent_diff = torch.zeros(
                (low.shape[0], 1, low.shape[2], low.shape[3]),
                device=low.device,
                dtype=low.dtype
            )
        else:
            if ent_diff.shape[-2:] != low.shape[-2:]:
                ent_diff = F.interpolate(
                    ent_diff,
                    size=low.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

        freq_low = freq_energy_fft(low)
        freq_high = freq_energy_fft(high)
        freq_ratio = freq_low / (freq_high + EPS)

        alpha = self.weight_predictor(
            torch.cat([ent_diff, freq_ratio], dim=1)
        )

        fused = alpha * high + (1.0 - alpha) * low
        fused = fused + self.refine(fused)

        return fused
