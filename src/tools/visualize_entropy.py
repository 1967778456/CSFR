import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.zoo.dfine.dfine import DFINE
from src.zoo.dfine.overlap_entropy import OverlapEntropySelector, MultiScaleEntropyEnhance

# ----------------------------- 配置 -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
image_dir = "data/images"  # 待可视化图片目录
save_dir = "visualizations/high_entropy_regions"
os.makedirs(save_dir, exist_ok=True)

preprocess = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

topk = 50  # 每张图片标记 top-k 高熵位置

# ----------------------------- 初始化模型 -----------------------------
# TODO: 请根据你自己的配置传入 backbone, encoder, decoder
model = DFINE(
    backbone=..., 
    encoder=..., 
    decoder=...
)
model = model.to(device).eval()

entropy_selector = OverlapEntropySelector(patch_size=3, stride=1).to(device)

# ----------------------------- 可视化工具 -----------------------------
def draw_entropy_boxes(img_pil, coords, patch_size=3, save_path=None):
    """在原图上标记高熵区域"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_pil)
    for (y, x) in coords:
        rect = patches.Rectangle(
            (x - patch_size // 2, y - patch_size // 2),
            patch_size, patch_size,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ----------------------------- 批量处理 -----------------------------
for fname in os.listdir(image_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path = os.path.join(image_dir, fname)
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        # 1. Backbone 特征
        feats = model.backbone(img_tensor)
        if isinstance(feats, list):
            feats = {f"level{i+1}": f for i, f in enumerate(feats)}
        elif isinstance(feats, torch.Tensor):
            feats = {"level1": feats}

        # 2. 单尺度熵
        entropy_maps_orig = {}
        for k, f in feats.items():
            if k in ["level1", "level2"]:
                entropy_maps_orig[k] = entropy_selector(f)

        # 3. 多尺度熵增强
        feats_list = [feats[k] for k in sorted(feats.keys())]
        if len(feats_list) >= 2:
            channels = [f.shape[1] for f in feats_list]
            entropy_enhancer = MultiScaleEntropyEnhance(channels).to(device)
            feats_enhanced = entropy_enhancer(feats_list)
        else:
            feats_enhanced = feats_list

        # 4. 可视化 top-k 高熵区域
        for i, k in enumerate(sorted(feats.keys())):
            if k not in entropy_maps_orig:
                continue
            entropy_orig = F.interpolate(entropy_maps_orig[k], size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)
            entropy_enh = F.interpolate(feats_enhanced[i].mean(1, keepdim=True), size=img_tensor.shape[-2:], mode='bilinear', align_corners=False)

            # 获取 top-k 坐标
            coords_orig = entropy_selector.topk_positions(entropy_orig)[0].cpu().numpy()
            coords_enh = entropy_selector.topk_positions(entropy_enh)[0].cpu().numpy()

            # 保存原始熵标记
            save_path_orig = os.path.join(save_dir, f"{fname.split('.')[0]}_{k}_topk_orig.png")
            draw_entropy_boxes(img_pil, coords_orig, patch_size=3, save_path=save_path_orig)

            # 保存增强熵标记
            save_path_enh = os.path.join(save_dir, f"{fname.split('.')[0]}_{k}_topk_enh.png")
            draw_entropy_boxes(img_pil, coords_enh, patch_size=3, save_path=save_path_enh)

            print(f"Saved: {save_path_orig}, {save_path_enh}")
