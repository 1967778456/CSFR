import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseAdaptiveFreqGate(nn.Module):
    """
    NoiseAdaptiveFreqGate
    ---------------------
    基于频率特征的轻量级噪声自适应门控模块，用于在特征图中自动抑制高噪声区域/通道。

    输入:  x ∈ R^{B×C×H×W}
    输出:  y ∈ R^{B×C×H×W}  (与输入同尺寸)

    设计要点:
    1. 频率特征: 通过低通近似 x_low 得到高频残差 x_high = x - x_low
    2. 空间噪声图: noise_spatial ≈ mean(|x_high|, dim=C)
    3. 通道噪声图: noise_channel ≈ mean(|x_high|, dim=H,W)
    4. 生成 [0,1] 的门控 (噪声越大, gate 越小), 对 x 做自适应抑制

    参数非常轻量，适合作为 plug-and-play 模块嵌入 backbone/FPN/PAN 等任意位置。
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        use_depthwise_blur: bool = False,
        blur_kernel_size: int = 3,
    ):
        """
        Args:
            channels       : 输入特征通道数 C
            reduction      : 通道门控中的中间维度压缩比例
            use_depthwise_blur: 是否使用 depthwise 可学习模糊卷积
                                 False 时默认使用 avg_pool 作为低通近似，更轻量
            blur_kernel_size: 低通卷积核大小 (仅在 use_depthwise_blur=True 时生效)
        """
        super().__init__()
        self.channels = channels
        self.reduction = max(1, reduction)
        self.use_depthwise_blur = use_depthwise_blur

        # 1) 低通近似 (频率特征的基础)
        if use_depthwise_blur:
            # 可学习低通 (depthwise conv)，初始化为平均滤波
            self.blur = nn.Conv2d(
                channels,
                channels,
                kernel_size=blur_kernel_size,
                padding=blur_kernel_size // 2,
                groups=channels,
                bias=False,
            )
            self._init_blur_as_mean(blur_kernel_size)
        else:
            # 使用 avg_pool 近似低通，无参数
            self.blur = None

        # 2) 空间噪声图 refinement (1 通道 → 1 通道)
        #    保持轻量，仅 1xConv(3x3)
        self.spatial_refine = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # 3) 通道噪声门控 (SE-like MLP，但用的是“噪声统计”)
        hidden_dim = max(8, channels // self.reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True),
        )

        # 4) 可学习缩放因子，用于控制噪声强度对 gate 的影响
        #    γ_spatial 控制空间维度上噪声 -> gate 的敏感度
        #    γ_channel 控制通道维度上噪声 -> gate 的敏感度
        self.gamma_spatial = nn.Parameter(torch.tensor(1.0))
        self.gamma_channel = nn.Parameter(torch.tensor(1.0))

    def _init_blur_as_mean(self, k: int):
        """将 depthwise blur 卷积初始化为平均滤波核。"""
        if not isinstance(self.blur, nn.Conv2d):
            return
        with torch.no_grad():
            weight = torch.zeros_like(self.blur.weight)  # [C,1,k,k]
            avg_kernel = torch.full((k, k), 1.0 / (k * k), dtype=weight.dtype)
            for c in range(weight.shape[0]):
                weight[c, 0] = avg_kernel
            self.blur.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            y: (B, C, H, W)
        """
        b, c, h, w = x.shape

        # -------------------------
        # 1) 低通近似 + 高频残差 (频率特征)
        # -------------------------
        if self.use_depthwise_blur and self.blur is not None:
            x_low = self.blur(x)               # 低频成分
        else:
            # avg_pool2d stride=1 模拟 3x3 低通
            x_low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        x_high = x - x_low                      # 高频残差 (包含纹理/噪声/边缘)
        x_high_abs = x_high.abs()               # 取绝对值方便统计

        # -------------------------
        # 2) 空间噪声图 noise_spatial: (B,1,H,W)
        # -------------------------
        # 对通道求平均, 得到每个空间位置的高频能量
        noise_spatial = x_high_abs.mean(dim=1, keepdim=True)  # [B,1,H,W]

        # 轻量 refinement (扩大感受野 + 平滑噪声估计)
        noise_spatial = self.spatial_refine(noise_spatial)    # [B,1,H,W]

        # 归一化 (防止数值过大)
        # 使用平均 + 标准差做简单标准化，避免除 0
        mean_s = noise_spatial.mean(dim=(2, 3), keepdim=True)
        std_s = noise_spatial.std(dim=(2, 3), keepdim=True) + 1e-6
        noise_spatial_norm = (noise_spatial - mean_s) / std_s

        # 噪声越大，希望 gate 越小:
        # gate_spatial = sigmoid( -γ_s * noise_norm )
        gate_spatial = torch.sigmoid(-self.gamma_spatial * noise_spatial_norm)  # [B,1,H,W]

        # -------------------------
        # 3) 通道噪声图 noise_channel: (B,C,1,1)
        # -------------------------
        noise_channel = x_high_abs.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]

        # 类似标准化
        mean_c = noise_channel.mean(dim=1, keepdim=True)
        std_c = noise_channel.std(dim=1, keepdim=True) + 1e-6
        noise_channel_norm = (noise_channel - mean_c) / std_c

        # 通道门控: 同样噪声越大 gate 越小
        channel_gate_raw = self.channel_mlp(noise_channel_norm)    # [B,C,1,1]
        gate_channel = torch.sigmoid(-self.gamma_channel * channel_gate_raw)  # [B,C,1,1]

        # -------------------------
        # 4) 空间 + 通道联合门控
        # -------------------------
        gate = gate_spatial * gate_channel         # [B,C,H,W] (广播)

        # -------------------------
        # 5) 应用到特征上
        # -------------------------
        # 方案 A: 直接乘 (最简单)
        # y = x * gate
        #
        # 方案 B: 带残差的“软抑制”，避免过度压制
        #         y = x * gate + x.detach() * (1 - gate.detach())
        #         这样对于梯度来说主要更新高噪声区域的 gate，而不会完全抹掉原始特征。
        #
        # 下面默认采用方案 A，如果你想更保守一点，可以改成方案 B。
        y = x * gate

        return y
