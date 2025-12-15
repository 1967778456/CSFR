# src/zoo/dfine/csfr/utils.py
import torch
import torch.nn.functional as F
from typing import List

EPS = 1e-8

__all__ = [
    "local_entropy",
    "local_variance",
    "cross_scale_entropy_diff",
]


# --------------------------------------------------
# 局部信息熵（空间域）
# --------------------------------------------------
def local_entropy(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    计算局部空间熵，用于衡量信息密度
    x: [B, C, H, W]
    return: [B, 1, H, W]
    """
    b, c, h, w = x.shape
    pad = kernel_size // 2

    patches = F.unfold(
        x, kernel_size=kernel_size, padding=pad, stride=1
    )  # [B, C*K*K, H*W]

    patches = patches.view(b, c, kernel_size * kernel_size, h * w)

    p = F.softmax(patches, dim=2) + EPS
    entropy = -torch.sum(p * torch.log(p), dim=2)  # [B, C, HW]

    entropy = entropy.mean(dim=1).view(b, 1, h, w)
    return entropy


# --------------------------------------------------
# 局部方差（噪声估计）
# --------------------------------------------------
def local_variance(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    基于局部方差的噪声估计
    x: [B, C, H, W]
    return: [B, 1, H, W]
    """
    pad = kernel_size // 2

    mean = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
    var = F.avg_pool2d((x - mean) ** 2, kernel_size, stride=1, padding=pad)

    var = var.mean(dim=1, keepdim=True)
    return var


# --------------------------------------------------
# 跨尺度熵差（In-CSFR 使用）
# --------------------------------------------------
def cross_scale_entropy_diff(feats: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    feats: 多尺度特征列表 [B,C,H,W]
    return: 每个尺度对应的熵差图 [B,1,H,W]
    """
    entropies = [local_entropy(f) for f in feats]
    diffs = []

    for i in range(len(feats)):
        curr = entropies[i]

        if i > 0:
            prev = F.interpolate(
                entropies[i - 1],
                size=curr.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            diff = torch.abs(curr - prev)
        else:
            next_ = F.interpolate(
                entropies[i + 1],
                size=curr.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            diff = torch.abs(curr - next_)

        diffs.append(diff)

    return diffs
