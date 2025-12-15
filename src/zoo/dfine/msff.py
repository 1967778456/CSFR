# msff.py (FINAL VERSION - SAFE + FULL LOSS)
import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import box_cxcywh_to_xyxy


# ============================================================
# Linear Attention（稳定 + 无非法 einsum）
# ============================================================
class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim)

        q = F.relu(q)
        k = F.relu(k)

        # KV = ∑ k ⊗ v
        kv = torch.einsum("bnhd,bnhp->bhdp", k, v)   # [B,H,D,D]

        # denom = q · (∑k)
        k_sum = k.sum(dim=1)                         # [B,H,D]
        denom = torch.einsum("bnhd,bhd->bnh", q, k_sum).clamp(min=1e-6)
        denom = denom.unsqueeze(-1)

        # context = q ⊗ KV
        ctx = torch.einsum("bnhd,bhdp->bnhp", q, kv)

        out = (ctx / denom).reshape(B, N, C)
        return self.norm(out + x)



# ============================================================
# Depthwise Separable Conv
# ============================================================
class DWConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, 3, padding=1, groups=c1)
        self.pw = nn.Conv2d(c1, c2, 1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))



# ============================================================
# MSFF（最终版 + 安全损失）
# ============================================================
class MSFF(nn.Module):
    def __init__(self, in_channels=(256,256,256), hidden_dim=256,
                 stride_p2=8, threshold=0.5, num_heads=4):
        super().__init__()

        C3, C4, C5 = in_channels
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.stride_p2 = stride_p2

        # -------------------------------
        # score / sigma heads
        # -------------------------------
        self.score_heads = nn.ModuleList([
            nn.Sequential(
                DWConv(ch, ch//2),
                nn.Conv2d(ch//2, 1, 1),
                nn.Sigmoid()
            ) for ch in in_channels
        ])

        self.sigma_heads = nn.ModuleList([
            nn.Sequential(
                DWConv(ch, ch//2),
                nn.Conv2d(ch//2, 1, 1),
                nn.Softplus()
            ) for ch in in_channels
        ])

        # -------------------------------
        # multi-scale alignment
        # -------------------------------
        self.c3_proj = DWConv(C3, hidden_dim)
        self.c4_up   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DWConv(C4, hidden_dim)
        )
        self.c5_up   = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            DWConv(C5, hidden_dim)
        )

        self.fuse = DWConv(hidden_dim, hidden_dim)

        # -------------------------------
        # Linear Attention
        # -------------------------------
        self.attn = LinearAttention(hidden_dim)
        self.out_norm = nn.BatchNorm2d(hidden_dim)



    # ============================================================
    # forward
    # ============================================================
    def forward(self, features, targets=None):
        C3, C4, C5 = features
        B, _, H, W = C3.shape
        device = C3.device

        refined = []
        init_scores = []
        sigmas = []

        # ----------------------------------------------------------
        # (1) soft gating（数值稳定）
        # ----------------------------------------------------------
        for f, sh, sg in zip([C3, C4, C5], self.score_heads, self.sigma_heads):

            p = sh(f)
            sigma = sg(f)

            score = p / (sigma + 1e-6)
            score = torch.clamp(score, 0, 3.0)

            gate = torch.sigmoid((score - self.threshold) * 10)

            selected = f * (0.5 + 0.5 * score * gate)

            refined.append(selected)
            init_scores.append(score)
            sigmas.append(sigma)

        # ----------------------------------------------------------
        # (2) multi-scale fusion
        # ----------------------------------------------------------
        c3 = self.c3_proj(refined[0])
        c4 = self.c4_up(refined[1])
        c5 = self.c5_up(refined[2])

        fused = self.fuse(c3 + c4 + c5)

        # ----------------------------------------------------------
        # (3) Linear Attention
        # ----------------------------------------------------------
        x = fused.flatten(2).transpose(1,2)
        x = self.attn(x)
        out = x.transpose(1,2).reshape(B, self.hidden_dim, H, W)
        out = self.out_norm(out + c3)

        # ----------------------------------------------------------
        # (4) loss（只在训练 + targets 存在时计算）
        # ----------------------------------------------------------
        loss = None
        if self.training and targets is not None:
            gt_boxes = self._convert_coco_targets(targets, device)
            loss = self._loss(init_scores, sigmas, gt_boxes, H, W)

        return out, loss



    # ============================================================
    # convert COCO targets → xyxy tensor[B,N,4]
    # ============================================================
    def _convert_coco_targets(self, targets, device):
        out = []
        for t in targets:
            boxes = t["boxes"]              # cxcywh normalized
            H, W = t["orig_size"].tolist()

            xyxy = box_cxcywh_to_xyxy(boxes)
            xyxy[:, [0,2]] *= W
            xyxy[:, [1,3]] *= H

            xyxy[:, 0] = xyxy[:, 0].clamp(0, W)
            xyxy[:, 2] = xyxy[:, 2].clamp(0, W)
            xyxy[:, 1] = xyxy[:, 1].clamp(0, H)
            xyxy[:, 3] = xyxy[:, 3].clamp(0, H)

            out.append(xyxy)

        return torch.nn.utils.rnn.pad_sequence(out, batch_first=True, padding_value=0.0).to(device)



    # ============================================================
    # MSFF loss（安全 + 不影响 DETR 逻辑）
    # ============================================================
    def _loss(self, init_scores, sigmas, gt_boxes, H, W):
        B = gt_boxes.shape[0]
        device = gt_boxes.device

        # -------------------------------
        # 1) build gt mask at C3 80×80
        # -------------------------------
        mask = torch.zeros((B,1,H,W), device=device)

        for b in range(B):
            for box in gt_boxes[b]:
                if box.sum() == 0:
                    continue
                x1,y1,x2,y2 = box.tolist()

                sx1 = int(x1 / self.stride_p2)
                sy1 = int(y1 / self.stride_p2)
                sx2 = int(x2 / self.stride_p2)
                sy2 = int(y2 / self.stride_p2)

                sx1 = max(0,sx1); sy1 = max(0,sy1)
                sx2 = min(W-1,sx2); sy2 = min(H-1,sy2)

                mask[b,0,sy1:sy2+1, sx1:sx2+1] = 1

        # -------------------------------
        # 2) L_score (Focal)
        # -------------------------------
        alpha=0.25
        gamma=2.0
        Ls = 0

        for s in init_scores:
            _,_,h,w = s.shape
            m = F.interpolate(mask, (h,w), mode="nearest")

            p = s.clamp(1e-6, 1-1e-6)

            pos = -alpha * (1-p)**gamma * m * torch.log(p)
            neg = -(1-alpha) * p**gamma * (1-m) * torch.log(1-p)

            Ls += (pos+neg).mean()

        Ls /= len(init_scores)

        # -------------------------------
        # 3) L_sigma (MSE)
        # -------------------------------
        Lsig = 0
        sigma_fg = 3
        sigma_bg = 10

        sigma_target = mask * sigma_fg + (1-mask) * sigma_bg

        for s in sigmas:
            _,_,h,w = s.shape
            t = F.interpolate(sigma_target, (h,w), mode="nearest")
            Lsig += F.mse_loss(s, t)

        Lsig /= len(sigmas)

        # -------------------------------
        # final
        # -------------------------------
        return Ls + 0.3 * Lsig
