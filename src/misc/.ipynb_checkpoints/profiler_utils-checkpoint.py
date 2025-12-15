"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from typing import Tuple

def stats(cfg, input_shape: Tuple = (1, 3, 640, 640)):
    """
    临时版本：完全关闭 FLOPs 和参数计算。
    返回占位信息，训练可以直接跑。
    """
    params = 0
    model_info = {"Model FLOPs: N/A   MACs: N/A   Params: N/A"}
    return params, model_info
