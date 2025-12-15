# src/zoo/dfine/csfr/pre_csfr.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import local_entropy, local_variance

__all__ = ["PreCSFR"]


# --------------------------------------------------
# FFT 频域分解（替代 torch_dct）
# --------------------------------------------------
class FFTDecompose(nn.Module):
    """
    使用 FFT 近似实现频域分解：
    - 低频：频谱中心区域
    - 高频：频谱边缘区域
    """
    def __init__(self, low_ratio: float = 0.25):
        """
        low_ratio: 低频区域占比（相对频谱尺寸）
        """
        super().__init__()
        self.low_ratio = low_ratio

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        return:
            low_freq:  [B, C, H, W]
            high_freq: [B, C, H, W]
        """
        b, c, h, w = x.shape

        # FFT
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

        # 构造低频 mask
        lh = int(h * self.low_ratio)
        lw = int(w * self.low_ratio)
        h_center, w_center = h // 2, w // 2

        mask = torch.zeros((h, w), device=x.device, dtype=x.dtype)
        mask[
            h_center - lh : h_center + lh,
            w_center - lw : w_center + lw
        ] = 1.0
        mask = mask[None, None, :, :]  # [1,1,H,W]

        # 低频 / 高频
        low_fft = fft_shift * mask
        high_fft = fft_shift * (1.0 - mask)

        # IFFT
        low = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1)).real
        high = torch.fft.ifft2(torch.fft.ifftshift(high_fft, dim=(-2, -1)), dim=(-2, -1)).real

        return low, high


# --------------------------------------------------
# Pre-CSFR 主模块
# --------------------------------------------------
class PreCSFR(nn.Module):
    """
    Pre-Encoder Cross-Scale Feature Reconstruction

    功能：
    - 熵驱动信息增强
    - FFT 频域结构分解（低频语义 / 高频结构）
    - 噪声自适应抑制
    """

    def __init__(
        self,
        channels: int,
        alpha: float = 1.0,   # 熵增强强度
        beta: float = 1.0,    # 噪声抑制强度
        low_freq_ratio: float = 0.25
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        # FFT 分解
        self.fft_decompose = FFTDecompose(low_ratio=low_freq_ratio)

        # 预归一化（统一语义空间）
        self.pre_norm = nn.GroupNorm(
            num_groups=max(1, channels // 4),
            num_channels=channels
        )

        # 高频结构增强
        self.high_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GroupNorm(max(1, channels // 4), channels),
            nn.GELU()
        )

        # 低频语义精炼
        self.low_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(max(1, channels // 4), channels),
            nn.GELU()
        )

        # 融合 + 输出精炼
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GroupNorm(max(1, channels // 4), channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        return: [B, C, H, W]
        """
        # 1. 统一特征空间
        x_norm = self.pre_norm(x)

        # 2. 信息熵 & 噪声估计
        entropy = local_entropy(x_norm)      # [B,1,H,W]
        noise = local_variance(x_norm)        # [B,1,H,W]

        # 3. FFT 频域分解
        low_freq, high_freq = self.fft_decompose(x_norm)

        # 4. 结构与语义分别增强
        low_feat = self.low_refine(low_freq)
        high_feat = self.high_enhance(high_freq)

        # 5. 融合重构
        recon = self.fuse(torch.cat([low_feat, high_feat], dim=1))

        # 6. 熵引导增强（信息密集区域）
        recon = recon * (1.0 + self.alpha * entropy)

        # 7. 噪声自适应抑制
        recon = recon * (1.0 - self.beta * noise)

        # 8. 残差连接
        out = x + recon

        return out
