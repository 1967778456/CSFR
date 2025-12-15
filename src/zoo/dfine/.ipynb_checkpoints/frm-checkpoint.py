import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_dct_mat(N, device):
    n = torch.arange(N, device=device).float().unsqueeze(0)  # [1, N]
    k = torch.arange(N, device=device).float().unsqueeze(1)  # [N, 1]
    mat = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
    mat[0] *= 1 / math.sqrt(2)
    mat = mat * math.sqrt(2 / N)
    return mat


def dct_2d(x):
    B, C, H, W = x.shape
    device = x.device
    D_h = build_dct_mat(H, device)
    D_w = build_dct_mat(W, device)

    x = x.reshape(B * C, H, W)
    x = torch.matmul(D_h, x)
    x = torch.matmul(x.transpose(-1, -2), D_w).transpose(-1, -2)
    return x.reshape(B, C, H, W)


def idct_2d(X):
    B, C, H, W = X.shape
    device = X.device
    D_h = build_dct_mat(H, device)
    D_w = build_dct_mat(W, device)

    X = X.reshape(B * C, H, W)
    X = torch.matmul(D_h.t(), X)
    X = torch.matmul(X.transpose(-1, -2), D_w.t()).transpose(-1, -2)
    return X.reshape(B, C, H, W)


class LearnableMaskFRM(nn.Module):
    """
    FRM v3: 频带 + 软阈值降噪 + 频段 SE + 空间频带选择（特征提纯版）

    1. DCT → Freq 频谱，并做归一化（避免不同图像/尺度幅值波动过大）
    2. 用半径 rr 构造低/中/高频三段 mask（互斥、非负、归一化）
    3. 在每个频带上做 band-wise soft-shrink（软阈值降噪）
    4. 各频带 IDCT → sp_low / sp_mid / sp_high（空间域表示）
    5. 频段 SE（band-wise SE）：根据三路 GAP 特征，产生 α_low/α_mid/α_high
    6. 空间频带选择：根据输入 x 的能量图生成 3 通道 A_low/A_mid/A_high
    7. 最终：Y = Σ_b α_b · A_b ⊙ sp_b，再经过共享 1×1 conv 融合，残差加回原特征
       out = x + λ · filter(cat([Y_low, Y_mid, Y_high]))
    """

    def __init__(self, channels, reduction=8, lambda_res=0.3,
                 tau_init=0.05):
        super().__init__()
        self.channels = channels
        self.lambda_res = lambda_res

        mid = max(8, channels // reduction)

        # 1) 三路频带联合建模的 filter（保持和你原来一致的 3C→mid→C）
        self.filter = nn.Sequential(
            nn.Conv2d(channels * 3, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        )

        # 2) 频带权重（低/中/高频全局系数），依旧用 softmax（+ sharpen）
        self.band_logits = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))  # [low, mid, high]

        # 3) band-wise soft-shrink 阈值（3 个频带，各一个标量）
        #    使用 softplus 保证阈值为正，初始值 tau_init
        init = math.log(math.exp(tau_init) - 1.0)
        self.soft_thr_logits = nn.Parameter(torch.tensor([init, init, init]))  # [3]

        # 4) 频段 SE（根据 sp_low/sp_mid/sp_high 的 GAP）
        #    输入维度为 3C，输出为 3（每个频带一个标量）
        self.band_se = nn.Sequential(
            nn.Linear(channels * 3, channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 3, bias=True)
        )

        # 5) 空间频带选择：由能量图 → 3 通道的空间权重图
        #    输入: [B,1,H,W]，输出: [B,3,H,W]
        self.spatial_gating = nn.Sequential(
            nn.Conv2d(1, mid, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 3, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    # ---------- 半径 & 频带 mask ----------
    def _build_radius_map(self, H, W, device):
        """
        构建归一化半径 rr ∈ [0, 1]，中心低频，边缘高频。
        """
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        rr = (yy.float() ** 2 + xx.float() ** 2)
        rr = rr / (rr.max() + 1e-6)  # 归一化到 [0, 1]
        rr = rr.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return rr

    def _build_band_masks(self, rr):
        """
        rr: [1, 1, H, W] ∈ [0, 1]
        返回:
            m_low, m_mid, m_high: [1, 1, H, W]，逐点相加约等于 1
        """
        # 基础非负 mask
        base_low = 1.0 - rr                     # 中心高，边缘低
        base_high = rr                          # 中心低，边缘高
        base_mid = 1.0 - torch.abs(2 * rr - 1)  # rr=0.5 处最大，在 0/1 处为 0

        base_low = torch.clamp(base_low, min=0.0)
        base_mid = torch.clamp(base_mid, min=0.0)
        base_high = torch.clamp(base_high, min=0.0)

        bases = torch.stack([base_low, base_mid, base_high], dim=0)  # [3, 1, 1, H, W]

        # sharpen band logits（和你之前一样，稍微拉开频带偏好）
        weights = F.softmax(self.band_logits * 2.0, dim=0)  # [3]
        weights = weights.view(3, 1, 1, 1, 1)

        masks = bases * weights  # [3, 1, 1, H, W]

        # 逐点归一化，防止能量发散（low+mid+high≈1）
        sum_masks = masks.sum(dim=0, keepdim=True)  # [1, 1, 1, H, W]
        masks = masks / (sum_masks + 1e-6)

        m_low, m_mid, m_high = masks[0], masks[1], masks[2]
        return m_low, m_mid, m_high

    # ---------- band-wise soft-shrink ----------
    def _soft_shrink_bands(self, F_low, F_mid, F_high):
        """
        对三个频带分别做 soft-shrink:
            y = sign(x) * relu(|x| - tau_b)
        tau_b 是每个频带一个学习到的标量阈值（>0）
        """
        # softplus 保证阈值大于 0
        tau = F.softplus(self.soft_thr_logits)  # [3] (positive)
        tau_low, tau_mid, tau_high = tau[0], tau[1], tau[2]

        # 低频：一般噪声少，阈值可以相对小，但这里交给网络自己学
        F_low_mag = F_low.abs()
        F_low = torch.sign(F_low) * torch.relu(F_low_mag - tau_low)

        # 中频：语义为主，也做同样形式的软阈值
        F_mid_mag = F_mid.abs()
        F_mid = torch.sign(F_mid) * torch.relu(F_mid_mag - tau_mid)

        # 高频：边缘 & 细节所在，对小目标很重要，网络可学得更小/更大阈值
        F_high_mag = F_high.abs()
        F_high = torch.sign(F_high) * torch.relu(F_high_mag - tau_high)

        return F_low, F_mid, F_high

    # ---------- forward ----------
    def forward(self, x):
        """
        输入:
            x: [B, C, H, W]

        输出:
            out: [B, C, H, W] (提纯后的特征)
        """
        B, C, H, W = x.shape
        device = x.device

        # 1) 频域变换
        Freq = dct_2d(x)  # [B, C, H, W]

        # 1.1) 频域归一化（防止尺度差异）
        Freq = Freq / (Freq.abs().amax(dim=(2, 3), keepdim=True) + 1e-6)

        # 2) 构建半径映射与频带 mask
        rr = self._build_radius_map(H, W, device)           # [1, 1, H, W]
        m_low, m_mid, m_high = self._build_band_masks(rr)   # [1, 1, H, W]

        # 3) 按频带划分频域特征（广播到 [B, C, H, W]）
        F_low = Freq * m_low
        F_mid = Freq * m_mid
        F_high = Freq * m_high

        # 3.1) band-wise soft-shrink 降噪
        F_low, F_mid, F_high = self._soft_shrink_bands(F_low, F_mid, F_high)

        # 4) 回到空间域
        sp_low = idct_2d(F_low)    # [B, C, H, W]
        sp_mid = idct_2d(F_mid)
        sp_high = idct_2d(F_high)

        # 5) 频段 SE: 根据三路 GAP 特征生成 α_low/mid/high
        #    每个 α_b 针对一个 band，是 [B,1,1,1] 形式的标量 gate
        g_low = F.adaptive_avg_pool2d(sp_low, 1).view(B, C)   # [B,C]
        g_mid = F.adaptive_avg_pool2d(sp_mid, 1).view(B, C)
        g_high = F.adaptive_avg_pool2d(sp_high, 1).view(B, C)

        g_cat = torch.cat([g_low, g_mid, g_high], dim=1)      # [B,3C]
        band_logits = self.band_se(g_cat)                     # [B,3]
        band_alpha = F.softmax(band_logits, dim=-1)           # [B,3]

        a_low = band_alpha[:, 0].view(B, 1, 1, 1)
        a_mid = band_alpha[:, 1].view(B, 1, 1, 1)
        a_high = band_alpha[:, 2].view(B, 1, 1, 1)

        # 6) 空间频带选择：根据输入特征 x 的能量图生成 [B,3,H,W] 的权重
        #    能量图：简单用通道均值的平方和（可以理解为一个“结构/纹理强度图”）
        energy = x.pow(2).mean(dim=1, keepdim=True)           # [B,1,H,W]

        # 归一化到 [0,1]
        with torch.no_grad():
            e_min = energy.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            e_max = energy.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            denom = (e_max - e_min).clamp(min=1e-6)
        energy_norm = (energy - e_min) / denom                # [B,1,H,W]

        spatial_weight = self.spatial_gating(energy_norm)     # [B,3,H,W], sigmoid
        A_low = spatial_weight[:, 0:1, :, :]                  # [B,1,H,W]
        A_mid = spatial_weight[:, 1:2, :, :]
        A_high = spatial_weight[:, 2:3, :, :]

        # 7) 组合 band-SE 和空间频带选择：
        #    Y_b = α_b · A_b ⊙ sp_b
        Y_low = a_low * (A_low * sp_low)      # [B,C,H,W] (A_low 会在通道维广播)
        Y_mid = a_mid * (A_mid * sp_mid)
        Y_high = a_high * (A_high * sp_high)

        # 8) 三路 concat → shared 1×1 conv filter → 残差输出
        sp_cat = torch.cat([Y_low, Y_mid, Y_high], dim=1)     # [B,3C,H,W]
        sp_enhance = self.filter(sp_cat)                      # [B,C,H,W]

        out = x + self.lambda_res * sp_enhance
        return out
