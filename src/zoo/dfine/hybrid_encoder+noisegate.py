"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.

Added: OverlapEntropy + AdaptiveAlpha + entropy-guided fusion in FPN/PAN
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .utils import get_activation

from .noise_gate import NoiseAdaptiveFreqGate   

__all__ = ["HybridEncoder"]


# -------------------------
# 熵计算模块
# -------------------------
class OverlapEntropy(nn.Module):
    def __init__(self, kernel_size=3, stride=1, eps=1e-8):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.unfold(x)  # [B, C*ks*ks, L]
        K = self.kernel_size * self.kernel_size
        patches = patches.view(B, C, K, -1)  # [B, C, K, L]

        p = F.softmax(patches, dim=2) + self.eps
        entropy = -torch.sum(p * torch.log(p), dim=2)  # [B, C, L]
        entropy = entropy.mean(dim=1)  # channel-average -> [B, L]
        entropy_map = entropy.view(B, 1, H, W)

        with torch.no_grad():
            e_det = entropy_map.detach()
            e_min = e_det.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            e_max = e_det.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            denom = (e_max - e_min).clamp(min=1e-6)
        entropy_map = (entropy_map - e_min) / denom
        return entropy_map



class AdaptiveAlpha(nn.Module):
    """空间 × 通道 双门控版本"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)

        # 通道门
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

        # 空间门
        self.spatial_fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        alpha_c = self.channel_fc(x)      # [B,C,1,1]
        alpha_s = self.spatial_fc(x)      # [B,1,H,W]
        return alpha_c * alpha_s          # 广播到 [B,C,H,W]


# -------------------------
# ConvNormLayer / ELAN / CSPLayer
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
# HybridEncoder
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # -------------------------
        # 输入投影
        # -------------------------
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    OrderedDict([
                        ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                        ("act", nn.ReLU(inplace=True))
                    ])
                )
            )

        # -------------------------
        # Transformer encoder
        # -------------------------
        encoder_layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout, enc_act)
        self.encoder = nn.ModuleList(
            [TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))]
        )

        # -------------------------
        # FPN top-down
        # -------------------------
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            self.fpn_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2))
            )

        # -------------------------
        # PAN bottom-up
        # -------------------------
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(SCDown(hidden_dim, hidden_dim, 3, 2))
            self.pan_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2))
            )

        # -------------------------
        # 熵增强 + 自适应 alpha
        # -------------------------
        num_fusion_layers = max(0, len(in_channels) - 1)
        self.entropy_module = OverlapEntropy(kernel_size=3, stride=1)
        self.alpha_fpn = nn.ModuleList([AdaptiveAlpha(hidden_dim * 2) for _ in range(num_fusion_layers)])
        self.alpha_pan = nn.ModuleList([AdaptiveAlpha(hidden_dim * 2) for _ in range(num_fusion_layers)])
        self.residual_weight = 0.1

        # ---------------------------------------------------
        # 频率噪声门控：每层使用不同 kernel 和不同开启策略
        # ---------------------------------------------------
        self.freq_gate_fpn = nn.ModuleList([
            NoiseAdaptiveFreqGate(hidden_dim * 2, use_depthwise_blur=True, blur_kernel_size=5),  # scale2
            NoiseAdaptiveFreqGate(hidden_dim * 2, use_depthwise_blur=True, blur_kernel_size=7),  # scale1
        ])
        
        self.freq_gate_pan = nn.ModuleList([
            NoiseAdaptiveFreqGate(hidden_dim * 2, use_depthwise_blur=True,  blur_kernel_size=5),  # scale2
            NoiseAdaptiveFreqGate(hidden_dim * 2, use_depthwise_blur=False, blur_kernel_size=3),  # scale3 (关闭)
        ])


    # -------------------------
    # forward
    # -------------------------
    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        # 输入投影
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Transformer 编码器
        for i, enc_idx in enumerate(self.use_encoder_idx):
            b, c, h, w = proj_feats[enc_idx].shape
            src = proj_feats[enc_idx].flatten(2).permute(0, 2, 1)
            pos_embed = None
            memory = self.encoder[i](src, pos_embed=pos_embed)
            proj_feats[enc_idx] = memory.permute(0, 2, 1).reshape(b, c, h, w).contiguous()

        # -------------------------
        # FPN top-down 融合
        # -------------------------
        inner_outs = [proj_feats[-1]]
        for idx in range(len(proj_feats) - 1, 0, -1):
            fused_id = len(proj_feats) - 1 - idx   # 0 → scale2, 1 → scale1
        
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[fused_id](feat_high)
            inner_outs[0] = feat_high
        
            upsample = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            fused = torch.cat([upsample, feat_low], dim=1)
        
            # 🎯 多尺度频率噪声抑制
            fused = self.freq_gate_fpn[fused_id](fused)
        
            # 熵增强
            entropy_map = self.entropy_module(fused)
            alpha = self.alpha_fpn[fused_id](fused)
            fused = fused * (1.0 + alpha * entropy_map)
        
            inner_outs[0] = self.fpn_blocks[fused_id](fused) + self.residual_weight * upsample


        # -------------------------
        # PAN bottom-up 融合
        # -------------------------
        pan_outs = [inner_outs[0]]
        for idx in range(len(proj_feats) - 1):
            
            feat_low = pan_outs[-1]
            feat_high = inner_outs[idx + 1] if idx + 1 < len(inner_outs) else proj_feats[idx + 1]
        
            down = self.downsample_convs[idx](feat_low)
            fused = torch.cat([down, feat_high], dim=1)
        
            # 🎯 多尺度频率噪声抑制
            fused = self.freq_gate_pan[idx](fused)
        
            # 熵增强
            entropy_map = self.entropy_module(fused)
            alpha = self.alpha_pan[idx](fused)
            fused = fused * (1.0 + alpha * entropy_map)
        
            pan_outs.append(self.pan_blocks[idx](fused) + self.residual_weight * down)


        return pan_outs
