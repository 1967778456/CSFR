"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
Modified by <Your Name>: Remove backbone output enhancement and OG_PTCA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register

__all__ = ["DFINE"]

@register()
class DFINE(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None, curr_epoch=None):
        # Type: <class 'dict'>
        # Keys: ['boxes', 'labels', 'image_id', 'image_path', 'keypoints', 'area', 'iscrowd', 'orig_size', 'idx']
        #   boxes: type=<class 'torch.Tensor'>, shape=torch.Size([6, 4]), values=tensor([[0.5637, 0.1684, 0.1400, 0.1123],
        #         [0.5584, 0.3149, 0.1737, 0.1070],
        #         [0.5468, 0.4070, 0.1863, 0.1123]], device='cuda:0')
        #   labels: type=<class 'torch.Tensor'>, shape=torch.Size([6]), values=tensor([8, 8, 8], device='cuda:0')
        #   image_id: type=<class 'torch.Tensor'>, shape=torch.Size([1]), values=tensor([218], device='cuda:0')
        #   image_path: type=<class 'str'>, value=/root/autodl-tmp/D-FINE/NWPU VHR-10/train/281.jpg
        #   keypoints: type=<class 'torch.Tensor'>, shape=torch.Size([6, 0, 3]), values=tensor([], device='cuda:0', size=(3, 0, 3))
        #   area: type=<class 'torch.Tensor'>, shape=torch.Size([6]), values=tensor([ 8512., 10065., 11328.], device='cuda:0')
        #   iscrowd: type=<class 'torch.Tensor'>, shape=torch.Size([6]), values=tensor([0, 0, 0], device='cuda:0')
        #   orig_size: type=<class 'torch.Tensor'>, shape=torch.Size([2]), values=tensor([950, 570], device='cuda:0')
        #   idx: type=<class 'torch.Tensor'>, shape=torch.Size([1]), values=tensor([218], device='cuda:0')
        
        feats = self.backbone(x)
        if isinstance(feats, list):
            feats = {f"level{i+1}": f for i, f in enumerate(feats)}
        elif isinstance(feats, torch.Tensor):
            feats = {"level1": feats}
        elif not isinstance(feats, dict):
            raise TypeError(f"Unexpected backbone output type: {type(feats)}")
        feats_list = [feats[k] for k in sorted(feats.keys())]


        # gt_boxes = []
        # if targets is not None:
        #     for t in targets:
        #         if "boxes" not in t:
        #             gt_boxes.append(None)
        #             continue
        
        #         boxes_norm = t["boxes"]      # [N,4] normalized cxcywh (0~1)
        #         h, w = t["orig_size"].tolist()
        
        #         # 转像素坐标
        #         boxes_abs = boxes_norm.clone()
        #         boxes_abs[:, 0] *= w   # cx
        #         boxes_abs[:, 1] *= h   # cy
        #         boxes_abs[:, 2] *= w   # width
        #         boxes_abs[:, 3] *= h   # height
        
        #         gt_boxes.append(boxes_abs)

        out = self.encoder(feats_list)
        out = self.decoder(out, targets)
        return out

    def deploy(self):
        """静态图推理优化"""
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
