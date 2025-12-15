"""
EDA 可视化批处理脚本
将 EDAVisualizer 保存的门控权重 (.npy 或图片) 生成彩色 heatmap 并保存
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

# 原始 EDA 保存目录
eda_dir = "outputs/eda"        # 这里改成你的 EDAVisualizer 保存目录
# 可视化输出目录
vis_dir = "outputs/eda_vis"
os.makedirs(vis_dir, exist_ok=True)

# 遍历目录
for fname in os.listdir(eda_dir):
    fpath = os.path.join(eda_dir, fname)
    
    # 加载门控数据
    if fname.endswith(".npy"):
        gate = np.load(fpath)
    elif fname.endswith((".png", ".jpg", ".jpeg")):
        gate = np.array(Image.open(fpath).convert("L")) / 255.0  # 灰度归一化
    else:
        continue

    # 如果是 1D 或 2D，直接 resize 到可视化大小
    if gate.ndim == 2:
        gate_resized = np.array(Image.fromarray(gate).resize((640, 640)))
    elif gate.ndim == 3:  # 多通道，取平均
        gate_resized = gate.mean(axis=0)
        gate_resized = np.array(Image.fromarray(gate_resized).resize((640, 640)))
    else:
        continue

    # 生成彩色 heatmap
    heatmap = cm.viridis(gate_resized)[:, :, :3]  # RGBA -> RGB

    # 保存 heatmap
    save_path = os.path.join(vis_dir, f"{os.path.splitext(fname)[0]}_heatmap.png")
    plt.imsave(save_path, heatmap)
    print(f"Saved heatmap: {save_path}")

print("✅ 所有门控权重可视化完成！")
