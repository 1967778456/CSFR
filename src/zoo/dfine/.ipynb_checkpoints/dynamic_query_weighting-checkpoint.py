# dynamic_query_weighting.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicQueryWeighting(nn.Module):
    """
    Dynamic Query Weighting (DQW)
    -------------------------------------
    基于熵图调整 DETR / DFINE 的 query embedding。
    高熵区域（信息丰富）得到更高的 query 权重，
    用于增强小目标检测能力。
    -------------------------------------
    参数:
        num_queries: decoder query 数量
        d_model: query embedding 维度
        lambda_q: 熵权缩放系数 (默认 0.5)
        mode: 权重融合模式 ('scale' or 'add')
    -------------------------------------
    输入:
        query_embed: [Nq, d_model] (nn.Parameter)
        entropy_maps: dict, {"level1": [B,1,H,W], ...}
    输出:
        加权后的 query_embed (tensor)
    """
    def __init__(self, num_queries, d_model, lambda_q=0.5, mode='scale'):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.lambda_q = lambda_q
        self.mode = mode

        # 将熵图映射到 query 空间的线性层
        self.entropy_proj = nn.Sequential(
            nn.Conv2d(1, d_model // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, d_model, 1)
        )

    def forward(self, query_embed, entropy_maps):
        """
        query_embed: [Nq, d_model]
        entropy_maps: dict(str->tensor)
        """
        if entropy_maps is None or len(entropy_maps) == 0:
            return query_embed

        # 选择用于小目标的熵图
        # 可根据你的 HGNetv2 输出层名调整
        e = None
        for key in ["level1", "p3", "s3", "small"]:
            if key in entropy_maps:
                e = entropy_maps[key]
                break
        if e is None:
            return query_embed

        B, _, H, W = e.shape

        # 熵特征 -> query 空间
        e_feat = self.entropy_proj(e)               # [B, d_model, H, W]
        e_feat = F.adaptive_avg_pool2d(e_feat, (1, 1)).view(B, self.d_model)
        e_weight = torch.sigmoid(e_feat.mean(0, keepdim=True))  # [1, d_model]

        # 根据模式融合
        if self.mode == 'scale':
            q_out = query_embed * (1 + self.lambda_q * e_weight)
        elif self.mode == 'add':
            q_out = query_embed + self.lambda_q * e_weight
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")

        return q_out
