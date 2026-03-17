"""
评估指标：帧级 mAP、calibrated mAP（cAP）、F1-score。

与原始实现数值严格对齐，兼容 numpy / torch 输入。
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional


def frame_level_map(
    probs: np.ndarray,
    labels: np.ndarray,
    with_bg: bool = False,
) -> dict[str, float | np.ndarray]:
    """计算帧级 mAP 和 calibrated mAP（cAP）。

    与原始 frame_level_map_n_cap 实现数值完全对齐。

    Args:
        probs:   [num_classes, num_frames]  预测概率
        labels:  [num_classes, num_frames]  二值标注（0/1）
        with_bg: 是否计入背景类（默认排除 index=0）

    Returns:
        {
            'map':       标量 mAP
            'cap':       标量 calibrated mAP
            'all_cls_ap':  [num_fg_classes] 逐类 AP
            'all_cls_acp': [num_fg_classes] 逐类 calibrated AP
        }
    """
    n_classes = labels.shape[0]
    all_cls_ap: list[float] = []
    all_cls_acp: list[float] = []

    from_index = 0 if with_bg else 1  # 默认跳过背景类

    for i in range(from_index, n_classes):
        this_cls_prob = probs[i, :]
        this_cls_gt = labels[i, :]

        n_pos = np.sum(this_cls_gt == 1)
        if n_pos == 0:
            all_cls_ap.append(float("nan"))
            all_cls_acp.append(float("nan"))
            continue

        # 背景帧数 / 动作帧数（用于 calibrated 权重）
        n_neg = np.sum(this_cls_gt == 0)
        w = n_neg / max(n_pos, 1)

        # 按预测分数降序排列
        indices = np.argsort(-this_cls_prob)
        tp, psum, cpsum = 0, 0.0, 0.0

        for k, idx in enumerate(indices):
            if this_cls_gt[idx] == 1:
                tp += 1
                wtp = w * tp
                fp = (k + 1) - tp
                psum += tp / (tp + fp)
                cpsum += wtp / (wtp + fp)

        this_cls_ap = psum / n_pos
        this_cls_acp = cpsum / n_pos

        all_cls_ap.append(this_cls_ap)
        all_cls_acp.append(this_cls_acp)

    # 对 nan 使用 nanmean（原实现）
    valid_ap = [v for v in all_cls_ap if not np.isnan(v)]
    valid_acp = [v for v in all_cls_acp if not np.isnan(v)]

    map_val = float(np.mean(valid_ap)) if valid_ap else 0.0
    cap_val = float(np.mean(valid_acp)) if valid_acp else 0.0

    return {
        "map": map_val,
        "cap": cap_val,
        "all_cls_ap": np.array(all_cls_ap),
        "all_cls_acp": np.array(all_cls_acp),
    }


def compute_f1_per_class(
    preds: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    num_classes: int,
    ignore_bg: bool = True,
) -> dict[str, float | np.ndarray]:
    """计算逐类别 F1-score 及整体 macro-F1。

    Args:
        preds:       [N] 预测类别 id
        labels:      [N] 真实类别 id
        num_classes: 类别总数（含背景）
        ignore_bg:   是否忽略背景类（id=0）

    Returns:
        {
            'macro_f1':    标量 macro-F1（不含背景）
            'per_class_f1': [num_fg_classes] 逐类 F1
            'per_class_acc': [num_classes] 逐类准确率
        }
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    preds = preds.flatten().astype(int)
    labels = labels.flatten().astype(int)

    per_class_f1: list[float] = []
    per_class_acc: list[float] = []

    start_cls = 1 if ignore_bg else 0

    for cls_id in range(num_classes):
        gt_mask = labels == cls_id
        pred_mask = preds == cls_id

        tp = int(np.logical_and(gt_mask, pred_mask).sum())
        fp = int(np.logical_and(~gt_mask, pred_mask).sum())
        fn = int(np.logical_and(gt_mask, ~pred_mask).sum())
        tn = int(np.logical_and(~gt_mask, ~pred_mask).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        acc = (tp + tn) / max(len(labels), 1)

        per_class_f1.append(f1)
        per_class_acc.append(acc)

    fg_f1 = [per_class_f1[i] for i in range(start_cls, num_classes)]
    macro_f1 = float(np.mean(fg_f1)) if fg_f1 else 0.0

    return {
        "macro_f1": macro_f1,
        "per_class_f1": np.array(per_class_f1[start_cls:]),
        "per_class_acc": np.array(per_class_acc),
    }
