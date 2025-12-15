"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Modified with CSFR (Cross-Scale Feature Reconstruction)
"""

import copy
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .utils import get_activation

# ===== CSFR imports =====
from .csfr.pre_csfr import PreCSFR
from .csfr.in_csfr import InCSFRBlock
from .csfr.post_csfr import PostCSFR
from .csfr.utils import cross_scale_entropy_diff

__all__ = ["HybridEncoder"]


# -------------------------
# ConvNormLayer / ELAN / CSPLayer (保留：原工程可能复用)
# -------------------------
class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        return self.act(y)


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=False, act="silu", bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[bottletype(hidden_channels, hidden_channels, act=get_activation(act)) for _ in range(num_blocks)]
        )
        self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class RepNCSPELAN4(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


# -------------------------
# Transformer Encoder
# -------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        if self.norm is not None:
            output = self.norm(output)
        return output


# -------------------------
# HybridEncoder (CSFR version, 对齐 D-FINE forward 流程)
# -------------------------
@register()
class HybridEncoder(nn.Module):
    __share__ = ["eval_spatial_size"]

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        # ===== CSFR switches =====
        use_pre_csfr=True,
        use_in_csfr=True,
        use_post_csfr=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.eval_spatial_size = eval_spatial_size

        self.use_pre_csfr = use_pre_csfr
        self.use_in_csfr = use_in_csfr
        self.use_post_csfr = use_post_csfr

        # -------------------------
        # Input projection
        # -------------------------
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                OrderedDict([
                    ("conv", nn.Conv2d(c, hidden_dim, 1, bias=False)),
                    ("norm", nn.BatchNorm2d(hidden_dim)),
                    ("act", nn.ReLU(inplace=True)),
                ])
            ) for c in in_channels
        ])

        # -------------------------
        # Pre-CSFR: 每个尺度一个（不改 backbone）
        # -------------------------
        if self.use_pre_csfr:
            self.pre_csfr = nn.ModuleList([PreCSFR(hidden_dim) for _ in range(len(in_channels))])

        # -------------------------
        # Transformer encoder (只对 use_encoder_idx 的尺度做)
        # -------------------------
        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout, enc_act)
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # -------------------------
        # In-CSFR: 替代传统 FPN/PAN 融合块
        # - 注意：这里每一层融合输出通道仍是 hidden_dim
        # -------------------------
        if self.use_in_csfr:
            num_fusion_layers = max(0, len(in_channels) - 1)
            self.fpn_blocks = nn.ModuleList([InCSFRBlock(hidden_dim, nhead) for _ in range(num_fusion_layers)])
            self.pan_blocks = nn.ModuleList([InCSFRBlock(hidden_dim, nhead) for _ in range(num_fusion_layers)])

        # -------------------------
        # Post-CSFR: 输出一致性（可选）
        # -------------------------
        if self.use_post_csfr:
            self.post_csfr = PostCSFR(
                channels_list=[hidden_dim] * len(in_channels),
                num_heads=nhead
            )

    # -------------------------
    # forward
    # -------------------------
    def forward(self, feats: List[torch.Tensor]):
        """
        feats: backbone 多尺度输出, e.g. [C3, C4, C5]
        return: pan_outs, 长度=3, 分辨率从高->低 (与 D-FINE 对齐)
        """
        assert len(feats) == len(self.in_channels)

        # 1) 输入投影到 hidden_dim
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # 2) Pre-CSFR（逐尺度重构）
        if self.use_pre_csfr:
            proj_feats = [self.pre_csfr[i](proj_feats[i]) for i in range(len(proj_feats))]

        # 3) Transformer 编码器（只对指定层）
        for i, enc_idx in enumerate(self.use_encoder_idx):
            b, c, h, w = proj_feats[enc_idx].shape
            src = proj_feats[enc_idx].flatten(2).permute(0, 2, 1)  # [B, HW, C]
            memory = self.encoder[i](src)
            proj_feats[enc_idx] = memory.permute(0, 2, 1).reshape(b, c, h, w).contiguous()

        # 4) 跨尺度熵差（按尺度对齐，给 In-CSFR 做动态融合因子）
        ent_diffs = None
        if self.use_in_csfr:
            ent_diffs = cross_scale_entropy_diff(proj_feats)  # List[B,1,H,W], 与每个尺度分辨率一致

        # =========================
        # 5) FPN top-down（输出 inner_outs：高分辨率 -> 低分辨率）
        # 关键点：要构建完整 pyramids，而不是只覆盖 inner_outs[0]
        # =========================
        inner_outs = [proj_feats[-1]]  # 先放最深层 (最低分辨率)
        L = len(proj_feats)

        # idx: L-1 -> 1
        for idx in range(L - 1, 0, -1):
            high = inner_outs[0]      # 当前 top-down 的“高层”（低分辨率）
            low = proj_feats[idx - 1] # 需要生成的“低层”（高分辨率）

            if self.use_in_csfr:
                # 注意：不要用关键字参数，避免签名不一致
                # InCSFRBlock 约定：输出分辨率=low_feat 的分辨率
                fused = self.fpn_blocks[L - 1 - idx](low, high, ent_diffs[idx - 1] if ent_diffs is not None else None)
            else:
                up = F.interpolate(high, size=low.shape[-2:], mode="nearest")
                fused = low + up

            # 生成的 fused 是更高分辨率的一层，插到最前面
            inner_outs.insert(0, fused)

        # 此时 inner_outs 长度=3，顺序：[P3, P4, P5]（高->低分辨率）

        # =========================
        # 6) PAN bottom-up（从高分辨率走向低分辨率）
        # 关键点：必须 downsample，再与对应层融合
        # =========================
        pan_outs = [inner_outs[0]]  # 从最高分辨率开始
        for idx in range(L - 1):
            curr = pan_outs[-1]       # 当前层（高分辨率）
            target = inner_outs[idx + 1]  # 下一层（低分辨率）

            if self.use_in_csfr:
                # 先下采样到 target 分辨率（严格 bottom-up）
                down = F.avg_pool2d(curr, kernel_size=2, stride=2)
                fused = self.pan_blocks[idx](down, target, ent_diffs[idx + 1] if ent_diffs is not None else None)
            else:
                down = F.avg_pool2d(curr, kernel_size=2, stride=2)
                fused = target + down

            pan_outs.append(fused)

        # pan_outs 长度=3，顺序：[N3, N4, N5]（高->低分辨率），与 D-FINE 对齐

        # 7) Post-CSFR（可选：输出一致性）
        if self.use_post_csfr:
            pan_outs = self.post_csfr(pan_outs)

        return pan_outs
