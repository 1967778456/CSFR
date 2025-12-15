"""
可视化 DFINE 模型中 MultiScaleEntropyEnhance 的可学习门控 (scale_gates)
Author: <Your Name>
Date: 2025-10
"""

import os
import torch
import matplotlib.pyplot as plt


def get_scale_gates(model):
    """
    从 DFINE 模型中提取 MultiScaleEntropyEnhance 模块的门控参数
    return: Tensor [num_scales]
    """
    gates = None
    for name, module in model.named_modules():
        if "entropy_enhance" in name and hasattr(module, "scale_gates"):
            gates = module.scale_gates.detach().cpu().sigmoid().numpy()
            print(f"[Info] Found scale_gates in {name}: {gates}")
            break
    if gates is None:
        raise RuntimeError("未在模型中找到 MultiScaleEntropyEnhance 的 scale_gates 参数。")
    return gates


def plot_gates_over_epochs(log_dir, save_path=None):
    """
    从训练日志中读取不同 epoch 保存的模型文件 (如 epoch_*.pth)
    并绘制 scale_gates 的变化曲线。
    """
    from src.zoo.dfine.dfine import DFINE  # 确保导入路径正确

    # 自动扫描日志目录下的权重文件
    ckpts = sorted(
        [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".pth")]
    )
    if not ckpts:
        raise FileNotFoundError(f"未在目录 {log_dir} 中找到模型权重文件 (*.pth)。")

    all_gates = []
    epochs = []

    for ckpt_path in ckpts:
        try:
            epoch_num = int(os.path.basename(ckpt_path).split("_")[-1].split(".")[0])
        except Exception:
            epoch_num = len(all_gates)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # 尝试加载模型
        model = DFINE(backbone=None, encoder=None, decoder=None)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Epoch {epoch_num}] Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        gates = get_scale_gates(model)
        all_gates.append(gates)
        epochs.append(epoch_num)

    all_gates = torch.tensor(all_gates).numpy()  # [num_epochs, num_scales]

    # 绘制
    plt.figure(figsize=(8, 5))
    for i in range(all_gates.shape[1]):
        plt.plot(epochs, all_gates[:, i], marker="o", label=f"scale_gate[{i+1}]")

    plt.title("Evolution of Multi-Scale Entropy Gates")
    plt.xlabel("Epoch")
    plt.ylabel("Gate value (after sigmoid)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] gate visualization: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize entropy scale gates")
    parser.add_argument(
        "--log_dir", type=str, required=True,
        help="训练权重目录路径，包含多个 epoch_*.pth"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="输出图像保存路径 (可选)"
    )
    args = parser.parse_args()

    plot_gates_over_epochs(args.log_dir, save_path=args.save)
