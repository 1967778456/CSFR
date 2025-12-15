# nade.py
# ---------------------------------------------------------
# NADE：Noise-Aware Dynamic Enhancement (FP32-Safe + SW-MSA + RPB)
# - 不依赖 torch_dct
# - DCT/IDCT 基于 FFT 实现，内部统一用 float32
# - 注意力使用 Window + Shifted Window（SW-MSA）
# - 带 Relative Position Bias
# - 噪声由频域检测 noise_mask 引导局部 LPE
# ---------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 0. Utility: 1D / 2D DCT-II & IDCT-II (基于 FFT 实现)
# ---------------------------------------------------------
def dct_1d(x: torch.Tensor) -> torch.Tensor:
    """
    沿最后一维计算 1D DCT-II（正交化），内部使用 float32。
    x: [..., N]
    return: [..., N] (float32)
    """
    x = x.float()
    N = x.shape[-1]

    # 构造镜像序列: [0..N-1, N-1..0]
    x_v = torch.cat([x, x.flip(dims=[-1])], dim=-1)  # [..., 2N]

    # FFT
    X = torch.fft.rfft(x_v, dim=-1)  # [..., N+1]

    k = torch.arange(N, device=x.device, dtype=x.dtype)
    W = torch.exp(-1j * torch.pi * k / (2 * N))  # [..., N]

    X = X[..., :N] * W  # 只取前 N 项
    return X.real  # DCT 结果为实数


def idct_1d(X: torch.Tensor) -> torch.Tensor:
    """
    沿最后一维计算 1D IDCT-II（DCT-II 的逆变换），内部使用 float32。
    X: [..., N]
    return: [..., N] (float32)
    """
    X = X.float()
    N = X.shape[-1]

    k = torch.arange(N, device=X.device, dtype=X.dtype)
    W = torch.exp(1j * torch.pi * k / (2 * N))

    Xc = X * W  # [..., N]

    # 构造共轭对称序列
    Xc_full = torch.cat([Xc, Xc[..., 1:-1].flip(dims=[-1])], dim=-1)  # [..., 2N]

    x = torch.fft.irfft(Xc_full, n=2 * N, dim=-1)  # [..., 2N]
    return x[..., :N]


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """
    2D DCT-II：对 H、W 维分别做 1D DCT-II
    x: [B,C,H,W]
    return: [B,C,H,W] (float32)
    """
    x = x.float()
    # 先对 W 维做 DCT
    x = dct_1d(x.transpose(-1, -2)).transpose(-1, -2)
    # 再对 H 维做 DCT
    x = dct_1d(x)
    return x


def idct_2d(x: torch.Tensor) -> torch.Tensor:
    """
    2D IDCT-II：对 H、W 维分别做 1D IDCT-II
    x: [B,C,H,W]
    return: [B,C,H,W] (float32)
    """
    x = x.float()
    # 先对 W 维做 IDCT
    x = idct_1d(x.transpose(-1, -2)).transpose(-1, -2)
    # 再对 H 维做 IDCT
    x = idct_1d(x)
    return x


# ---------------------------------------------------------
# 1. Frequency Noise Detector
# ---------------------------------------------------------
class FreqNoiseDetector(nn.Module):
    """
    频域噪声探测：
      - 高频能量弱 & 分散 → 噪声概率高
      - 高频集中 → 有用细节
    输出 noise_mask ∈ [0,1]，1 表示更像噪声
    """

    def __init__(self, win_size: int = 3):
        super().__init__()
        pad = win_size // 2

        self.energy_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=win_size, padding=pad, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.noise_conf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W] (float32 或 float16 均可，内部会转为 float32)
        return: [B,1,H,W] (float32)
        """
        x = x.float()
        B, C, H, W = x.shape

        # 1) 频域能量
        freq = torch.abs(dct_2d(x))  # [B,C,H,W]

        # 2) 归一化到 [0,1]
        min_v = freq.amin(dim=[2, 3], keepdim=True)
        max_v = freq.amax(dim=[2, 3], keepdim=True)
        freq_norm = (freq - min_v) / (max_v + 1e-6)

        # 3) 通道平均 → [B,1,H,W] → 局部能量
        local_energy = self.energy_conv(freq_norm.mean(dim=1, keepdim=True))

        # 4) 全局噪声评分
        noise_global = self.noise_conf(local_energy)  # [B,1]

        # 5) 局部噪声评分（能量越低越像噪声）
        noise_local = torch.sigmoid(1 - local_energy)  # [B,1,H,W]

        # 6) 融合：局部 × 全局
        noise_mask = noise_local * noise_global.view(B, 1, 1, 1)  # [B,1,H,W]
        return noise_mask


# ---------------------------------------------------------
# 2. Window Attention with Relative Position Bias
# ---------------------------------------------------------
class WindowAttentionWithRPB(nn.Module):
    """
    Window Multi-head Self Attention (W-MSA) + Relative Position Bias
    x: [B*nW, Nw, C]   Nw = window_size * window_size
    """

    def __init__(self, dim, window_size=12, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # ------------------------------
        # Relative Position Bias Table
        # 大小：(2*ws-1) * (2*ws-1) * num_heads
        # ------------------------------
        ws = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )  # [L, num_heads]

        # 生成每个 token pair 的 index（Nw x Nw）
        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, ws, ws]
        coords_flatten = torch.flatten(coords, 1)  # [2, Nw]

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Nw, Nw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Nw, Nw, 2]

        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        relative_position_index = relative_coords.sum(-1)  # [Nw, Nw]

        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*nW, Nw, C]
        """
        BnW, Nw, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(BnW, Nw, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # [3, BnW, heads, Nw, head_dim]
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [BnW, heads, Nw, Nw]

        # 相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(Nw, Nw, -1)  # [Nw, Nw, heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # [heads, Nw, Nw]
        attn = attn + relative_position_bias.unsqueeze(0)  # [1,heads,Nw,Nw] 广播到 BnW

        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(BnW, Nw, C)
        out = self.proj(out)
        return out


class DynamicAttention(nn.Module):
    """
    自适应 SW-MSA + 相对位置偏置 + 噪声引导 LPE
    完全兼容任意 H×W（如 72, 80, 96, 112 ...）
    """

    def __init__(self, dim, num_heads=8, max_window_size=12):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_window_size = max_window_size

        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # qkv
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # LPE
        self.lpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.lpe_scale = nn.Parameter(torch.ones(1))

        # 用于缓存 bias table，避免每次 forward 重建
        self._cache = {}

    # -------------------------------------------------------------
    # 自动找到合适的 window_size（必须整除 H 和 W）
    # -------------------------------------------------------------
    def get_adaptive_window_size(self, H, W):
        factors_H = [f for f in range(1, H+1) if H % f == 0]
        factors_W = [f for f in range(1, W+1) if W % f == 0]
        common = [f for f in factors_H if f in factors_W and f <= self.max_window_size]
        return max(common) if len(common) > 0 else min(H, W)

    # -------------------------------------------------------------
    # 相对位置偏置
    # -------------------------------------------------------------
    def build_relative_position_bias(self, ws, device):
        if ws in self._cache:
            return self._cache[ws]

        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws)
        ))  # [2, ws, ws]

        coords_flat = coords.reshape(2, -1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # [2, N, N]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()

        # 偏置表大小：(2*ws-1)*(2*ws-1)
        bias_size = 2 * ws - 1
        relative_position_bias_table = nn.Parameter(
            torch.zeros((bias_size * bias_size, self.num_heads))
        )

        # 偏置索引
        relative_coords = rel_coords + (ws - 1)
        relative_position_index = relative_coords[..., 0] * (2 * ws - 1) + relative_coords[..., 1]

        # 缓存
        self._cache[ws] = (
            relative_position_bias_table.to(device),
            relative_position_index.to(device)
        )

        return self._cache[ws]

    # -------------------------------------------------------------
    # Window Partition
    # -------------------------------------------------------------
    def window_partition(self, x, ws):
        B, H, W, C = x.shape
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        return x

    def window_reverse(self, windows, ws, B, H, W):
        x = windows.reshape(B, H // ws, W // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
        return x

    # -------------------------------------------------------------
    # SW-MSA Forward
    # -------------------------------------------------------------
    def forward(self, x_seq, noise_mask):
        B, N, C = x_seq.shape
        H = W = int(N ** 0.5)

        ws = self.get_adaptive_window_size(H, W)  # ⭐ 关键：自动 window_size

        x = x_seq.reshape(B, H, W, C)

        # --- 相对位置偏置 ---
        rel_pos_bias_table, rel_pos_index = self.build_relative_position_bias(ws, x.device)

        # --- partition windows ---
        windows = self.window_partition(x, ws)  # [B*nW, ws*ws, C]

        # qkv
        qkv = self.qkv(windows).reshape(
            windows.shape[0], ws * ws, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # [BnW, heads, Nw, head_dim]

        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [BnW, h, Nw, Nw]

        # 加相对位置偏置
        relative_bias = rel_pos_bias_table[rel_pos_index.reshape(-1)].reshape(
            ws * ws, ws * ws, self.num_heads
        ).permute(2, 0, 1)  # [h, Nw, Nw]

        attn = attn + relative_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        out = self.proj(out)

        # --- reverse windows ---
        x_attn = self.window_reverse(out, ws, B, H, W)

        # --- 噪声引导 LPE ---
        x_lpe = self.lpe(x.permute(0, 3, 1, 2)) * self.lpe_scale * noise_mask
        x_lpe = x_lpe.permute(0, 2, 3, 1)

        # --- 融合 ---
        x_out = x_attn + x_lpe

        return x_out.reshape(B, N, C)

# ---------------------------------------------------------
# 4. NADE: Noise-Aware Dynamic Enhancement
#    ★ 内部强制 FP32 + 禁用 autocast，外部 dtype 不变
# ---------------------------------------------------------
class NADE(nn.Module):
    """
    NADE 全流程（建议只在 C3 / 1/8 尺度上使用）：
      1. FreqNoiseDetector：频域识别噪声区域 → noise_mask
      2. DynamicAttention(SW-MSA+RPB)：噪声区域弱化全局注意力 + 强化局部 LPE
      3. 频域 soft gate：抑制不稳定高频，保留有用高频
      4. 1x1 conv refine + 残差融合，输出增强特征
    """

    def __init__(self, in_channels: int, num_heads: int = 8, window_size: int = 12):
        super().__init__()

        self.noise_detector = FreqNoiseDetector()
        self.dynamic_attn = DynamicAttention(
            dim=in_channels,
            num_heads=num_heads,
        )

        self.freq_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W] 可以是 float16 / float32
        return: [B,C,H,W]，dtype 与输入一致
        """
        orig_dtype = x.dtype

        # ⭐ 关闭 AMP，内部全用 float32，避免 FFT 在 FP16 非 2 次幂尺寸下的限制
        with torch.cuda.amp.autocast(enabled=False):
            x_fp32 = x.float()
            B, C, H, W = x_fp32.shape
            N = H * W

            # ---- Step1: 频域噪声检测 ----
            noise_mask = self.noise_detector(x_fp32)  # [B,1,H,W]

            # ---- Step2: 动态注意力增强（SW-MSA + RPB + LPE）----
            x_seq = x_fp32.flatten(2).transpose(1, 2)  # [B,N,C]
            x_attn = self.dynamic_attn(x_seq, noise_mask)
            x_attn = x_attn.transpose(1, 2).reshape(B, C, H, W)  # [B,C,H,W]

            # ---- Step3: 频域 soft gate 高频 ----
            freq = dct_2d(x_attn)                                  # [B,C,H,W]
            freq_mean = freq.mean(dim=[2, 3], keepdim=True)        # [B,C,1,1]

            # soft 门控：幅值高于均值的高频保留更多
            gate = torch.sigmoid(5 * (torch.abs(freq) - freq_mean))  # [B,C,H,W]
            freq_refined = freq * gate

            x_freq = idct_2d(freq_refined)  # [B,C,H,W]

            # ---- Step4: refine + 残差 ----
            out_fp32 = self.freq_refine(x_fp32 + x_freq)  # [B,C,H,W]

        # 转回原始 dtype（例如 float16）
        return out_fp32.to(orig_dtype)
