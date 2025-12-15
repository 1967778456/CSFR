"""
位置先验（PosMap） + 熵增强模块（无蒸馏版）
核心设计思想：
  1) 熵图 Entropy：仅由特征计算，粗粒度定位“值得增强的区域”
  2) 学生位置图 PosMap：告诉网络哪里更可能是目标区域
  3) 自适应双门控 Alpha：
       - 通道门 alpha_c：选择哪些通道需要增强（类/语义相关）
       - 空间门 alpha_s：在熵高的区域里，筛出真正的目标像素
     Alpha 只依赖 feature + posmap，不依赖 entropy，本质是一个“目标感知的注意力图”
  4) 最终增强（在 FPN / PAN 中使用）：
       fused_enh = fused * (1 + entropy_map * alpha)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation


# ================================================================
# 1. 纯熵图 OverlapEntropy（不再注入位置先验）
# ================================================================
class OverlapEntropyWithPos(nn.Module):
    """
    虽然名字里带 WithPos，为了兼容 HybridEncoder 不改类名，
    但这里已经不再使用 pos_map，熵图只由特征 x 计算。
    """
    def __init__(self, kernel_size=3, stride=1, eps=1e-8, num_scales=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.unfold = nn.Unfold(
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )

    def forward(self, x, pos_map=None, scale_idx=0):
        """
        x: [B, C, H, W]
        返回：entropy_map [B, 1, H, W]，归一化到 [0,1]
        """
        B, C, H, W = x.shape

        # unfold → 局部 patch
        patches = self.unfold(x)                   # [B, C*K, L]
        K = self.kernel_size * self.kernel_size
        patches = patches.view(B, C, K, -1)        # [B, C, K, L]

        # patch 内 softmax 当作概率
        p = F.softmax(patches, dim=2) + self.eps   # [B, C, K, L]
        entropy = -torch.sum(p * torch.log(p), dim=2)   # [B, C, L]
        entropy = entropy.mean(dim=1)                    # [B, L]
        entropy = entropy.view(B, 1, H, W)               # [B, 1, H, W]

        # batch 内归一化 → [0,1]
        with torch.no_grad():
            emin = entropy.amin(dim=(1, 2, 3), keepdim=True)
            emax = entropy.amax(dim=(1, 2, 3), keepdim=True)
        entropy = (entropy - emin) / (emax - emin + self.eps)

        # 不再加入任何 pos_map
        return entropy


# ================================================================
# 2. 自适应 Alpha（通道 + 空间），只看 feat + posmap
# ================================================================
class AdaptiveAlphaWithPos(nn.Module):
    """
    Alpha 的定位作用：
      - 通道门 alpha_c：告诉“增强哪些语义通道”
      - 空间门 alpha_s：在粗粒度 entropy 区域中，挑出更像目标的像素
    Alpha 不使用 entropy，只使用 feature + posmap。
    最终在 HybridEncoder 中使用： fused_enh = fused * (1 + entropy * alpha)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)

        # ---------- 通道门：全局池化 + MLP ----------
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()  # [B, C, 1, 1]
        )

        # ---------- 空间门：posmap 引导 + feature 细化 ----------
        # posmap 先做一个小 conv，得到“目标注意力图”
        self.pos_conv = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()   # [B, 1, H, W]
        )

        # 再将 feature + pos_attn 融合，生成最终的 spatial gate
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()   # [B, 1, H, W]
        )

        # 单尺度增强分支：结构简单一点，用于 pre-fusion
        self.single_scale_conv = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()   # [B, C, H, W]
        )

    # -----------------------------
    # 融合前：单尺度增强（只看特征 + posmap）
    # -----------------------------
    def forward_single_scale(self, x, pos_map):
        """
        x: [B, C, H, W]
        pos_map: [B, 1, H', W'] or None
        """
        if pos_map is None:
            return x
        pos_map_resized = F.interpolate(
            pos_map, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        alpha_single = self.single_scale_conv(
            torch.cat([x, pos_map_resized], dim=1)
        )  # [B, C, H, W]
        # 这里只做轻微增强，避免破坏 backbone 语义
        return x * (1.0 + 0.2 * alpha_single)

    # -----------------------------
    # 融合时：双门控（通道 + 空间），只用 feat + posmap
    # -----------------------------
    def forward_fusion(self, x, entropy_map, pos_map):
        """
        x: [B, C, H, W]
        entropy_map: [B, 1, H, W]（这里不参与 alpha 计算，只在外部用于乘）
        pos_map: [B, 1, H', W'] or None
        返回：alpha [B, 1, H, W]，在外部与 entropy 相乘使用
        """
        B, C, H, W = x.shape

        # 通道门：与熵无关，主要看类别/语义
        alpha_c = self.channel_fc(x)  # [B, C, 1, 1]

        # 空间门：由 posmap 指引
        if pos_map is None:
            # 没有 posmap 的情况下，退化为“通道注意力 + feature 空间注意力”
            dummy_pos = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
            pos_attn = self.pos_conv(dummy_pos)  # [B,1,H,W]
        else:
            pos_resized = F.interpolate(
                pos_map, size=(H, W), mode="bilinear", align_corners=False
            )
            pos_attn = self.pos_conv(pos_resized)  # [B,1,H,W]

        # 特征 + pos_attn → 空间 gate
        spatial_input = torch.cat([x, pos_attn], dim=1)   # [B, C+1, H, W]
        alpha_s = self.spatial_conv(spatial_input)        # [B, 1, H, W]

        # 将通道门广播到空间： [B,C,1,1] → [B,C,H,W]
        alpha_c_map = alpha_c.expand(-1, -1, H, W)        # [B, C, H, W]
        # 再压成 1 通道（取平均），得到 [B,1,H,W] 的通道注意力
        alpha_c_spatial = alpha_c_map.mean(dim=1, keepdim=True)  # [B,1,H,W]

        # 最终 alpha：通道感知 × 空间感知
        alpha = alpha_c_spatial * alpha_s  # [B,1,H,W]

        return alpha  # 后续在 HybridEncoder 中与 entropy_map 相乘使用

        
    def forward(self, x, entropy_map=None, pos_map=None):
        return self.forward_fusion(x, entropy_map, pos_map)

# ================================================================
# 3. 学生位置先验（主干与之前版本兼容，只是供 alpha 引导）
# ================================================================
class StudentPosHead(nn.Module):
    """
    输出：每个 scale 的概率位置图 [B,1,H,W]
    监督：center + ring（由 compute_label_loss 使用）
    """
    def __init__(self, feat_channels_list, mid_scale_idx=1, mid_stride=16):
        super().__init__()
        self.mid_scale_idx = mid_scale_idx
        self.mid_stride = mid_stride
        self.num_scales = len(feat_channels_list)

        mid_ch = feat_channels_list[mid_scale_idx]
        self.mid_base = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 4, 1, 1)
        )

        self.scale_refine = nn.ModuleList()
        for ch in feat_channels_list:
            self.scale_refine.append(
                nn.Sequential(
                    nn.Conv2d(ch + 1, ch // 8, 1, bias=False),
                    nn.BatchNorm2d(ch // 8),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch // 8, 1, 3, padding=1),
                )
            )

        self.sigmoid = nn.Sigmoid()

    # ============================================================
    # 内部：生成中心 + ring 高斯图，用于标签监督（非必须用于增强）
    # ============================================================
    def _make_gaussian_and_ring(self, H, W, boxes, stride, device):
        ys = torch.arange(H, device=device).view(H, 1)
        xs = torch.arange(W, device=device).view(1, W)

        center_map = torch.zeros(1, H, W, device=device)
        ring_map = torch.zeros(1, H, W, device=device)

        for box in boxes:
            if box.sum() < 1e-6:
                continue

            # 转到 feature map 坐标（你的 gt_boxes 是 cx,cy,w,h 像素）
            cx, cy, bw, bh = box
            x1 = (cx - bw / 2) / stride
            x2 = (cx + bw / 2) / stride
            y1 = (cy - bh / 2) / stride
            y2 = (cy + bh / 2) / stride

            x1 = x1.clamp(0, W - 1)
            x2 = x2.clamp(0, W - 1)
            y1 = y1.clamp(0, H - 1)
            y2 = y2.clamp(0, H - 1)

            cx_f = (x1 + x2) / 2
            cy_f = (y1 + y2) / 2
            w = (x2 - x1 + 1).clamp(min=2.0)
            h = (y2 - y1 + 1).clamp(min=2.0)

            area = w * h

            # 小目标：只用中心高斯
            if area < 20 * 20:
                sigma_x = w
                sigma_y = h
                g = torch.exp(-(((xs - cx_f) ** 2) / (2 * sigma_x ** 2) +
                                ((ys - cy_f) ** 2) / (2 * sigma_y ** 2)))
                center_map = torch.max(center_map, g)
                continue

            # 大目标：中心 + 环形（轮廓附近）
            sigma_x = w
            sigma_y = h
            inner = torch.exp(-(((xs - cx_f) ** 2) / (2 * sigma_x ** 2) +
                                ((ys - cy_f) ** 2) / (2 * sigma_y ** 2)))
            outer = torch.exp(-(((xs - cx_f) ** 2) / (2 * (1.8 * sigma_x) ** 2) +
                                ((ys - cy_f) ** 2) / (2 * (1.8 * sigma_y) ** 2)))
            ring = (outer - inner).clamp(0, 1)

            center_map = torch.max(center_map, inner)
            ring_map = torch.max(ring_map, ring)

        return center_map, ring_map

    # ============================================================
    # 🔷 GT supervision: 生成中心图 + 边缘环图（供 loss 用）
    # ============================================================
    def generate_gt_pos_maps(self, gt_boxes, feats_list):
        """
        gt_boxes: List[Tensor], 每个 [Ni,4]，像素坐标 cx,cy,w,h
        feats_list: 多尺度特征
        返回：List[Tensor]，每个尺度 [B,2,H,W]（0: center, 1: ring）
        """
        gt_maps = []
        device = feats_list[0].device

        for s in range(self.num_scales):
            feat = feats_list[s]
            B, _, H, W = feat.shape
            stride = self.mid_stride // (2 ** (self.mid_scale_idx - s))

            maps = torch.zeros(B, 2, H, W, device=device)

            for b in range(B):
                if gt_boxes[b] is None or gt_boxes[b].numel() == 0:
                    continue
                c_map, r_map = self._make_gaussian_and_ring(
                    H, W, gt_boxes[b], stride, device
                )
                maps[b, 0] = c_map
                maps[b, 1] = r_map

            gt_maps.append(maps.clamp(0, 1))

        return gt_maps

    # ============================================================
    # 🔷 Forward: 生成学生位置图（供 alpha 使用）
    # ============================================================
    def forward(self, feats_list):
        """
        feats_list: List[Tensor] 多尺度特征
        返回： List[Tensor]，每个 [B,1,H,W]（概率图）
        """
        mid_feat = feats_list[self.mid_scale_idx]
        mid_pos = self.mid_base(mid_feat)  # [B,1,H_mid,W_mid]

        pos_maps = []
        for s, feat in enumerate(feats_list):
            H, W = feat.shape[2:]
            pos_interp = F.interpolate(
                mid_pos, size=(H, W), mode="bilinear", align_corners=False
            )
            refine = self.scale_refine[s](torch.cat([feat, pos_interp], dim=1))
            pos = 0.7 * pos_interp + 0.3 * refine
            pos_maps.append(self.sigmoid(pos))  # [B,1,H,W]
        return pos_maps


# ================================================================
# 4. 标签监督损失（中心 + 环形边缘）
# ================================================================
def compute_label_loss(student_maps, gt_maps, scale_weights):
    """
    student_maps: List[Tensor], 每个 [B,1,H,W]
    gt_maps:      List[Tensor], 每个 [B,2,H,W] (center, ring)
    scale_weights: List[float]，每个尺度的 loss 权重，例如 [0.25, 0.5, 0.25]
    """
    device = student_maps[0].device
    total = torch.tensor(0.0, device=device)
    eps = 1e-6

    # 转成 tensor 方便在 GPU 上做
    scale_weights_t = torch.tensor(
        scale_weights, device=device, dtype=torch.float32
    )

    for s_idx, (s_map, g_map) in enumerate(zip(student_maps, gt_maps)):
        if g_map is None:
            continue

        center = g_map[:, 0:1]  # [B,1,H,W]
        ring = g_map[:, 1:2]    # [B,1,H,W]

        s = s_map.clamp(eps, 1 - eps)

        # BCE for center
        logits = torch.logit(s, eps=eps)
        bce_center = F.binary_cross_entropy_with_logits(
            logits.float(), center.float()
        )

        # L1 for ring
        ring_loss = F.l1_loss(s.float(), ring.float())

        w = scale_weights_t[s_idx]
        total = total + w * (bce_center + 0.5 * ring_loss)

    return total
