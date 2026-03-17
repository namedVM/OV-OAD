"""
OV-OAD 独立分布式评估脚本（基于 Hugging Face Accelerate）

accelerate 自动处理：分布式推理 gather、设备迁移、混合精度。
无需手动编写 all_gather / DDP 包装。

用法：
    # 单卡
    python evaluate.py --config configs/train.yml \
                       --checkpoint outputs/.../checkpoints/best_map

    # 多卡
    accelerate launch evaluate.py \
        --config configs/train.yml \
        --checkpoint outputs/.../checkpoints/best_map \
        --output-dir results/eval_run

    # torchrun 方式
    torchrun --nproc_per_node=8 evaluate.py \
        --config configs/train.yml \
        --checkpoint outputs/.../checkpoints/step_0010000
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed

from ovoad.datasets import build_dataloaders
from ovoad.models import build_model
from ovoad.utils import compute_f1_per_class, frame_level_map, setup_logging

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# 参数解析
# ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OV-OAD 分布式评估（Accelerate）")
    parser.add_argument("--config",          type=str, required=True,  help="训练配置文件（YAML）")
    parser.add_argument("--checkpoint",      type=str, required=True,  help="accelerate checkpoint 目录")
    parser.add_argument("--output-dir",      type=str, default="",     help="评估结果目录（默认：checkpoint 同级）")
    parser.add_argument("--data-dir",        type=str,                 help="覆盖：.pth 数据目录")
    parser.add_argument("--metadata-csv",    type=str,                 help="覆盖：metadata.csv 路径")
    parser.add_argument("--batch-size",      type=int,                 help="覆盖：评估 batch size")
    parser.add_argument("--num-workers",     type=int, default=4,      help="DataLoader worker 数")
    parser.add_argument("--split",           type=str, default="val",  choices=["val", "train"])
    parser.add_argument("--with-bg",         action="store_true",      help="mAP 是否含背景类")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--save-preds",      action="store_true", default=True, help="保存预测矩阵")
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────
# 评估核心
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: list[str],
    with_bg: bool = False,
    split: str = "val",
) -> dict[str, Any]:
    """对数据集进行完整推理并计算帧级指标。

    Args:
        accelerator: Accelerate 实例（自动处理分布式 gather）
        model:       已经过 accelerator.prepare 的模型
        loader:      已经过 accelerator.prepare 的 DataLoader
        class_names: 类别名列表（index 即 id）
        with_bg:     mAP 是否含背景类
        split:       数据集划分名（用于日志）

    Returns:
        包含 probs、labels、指标的字典
    """
    model.eval()
    num_classes = len(class_names)

    all_probs:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    if accelerator.is_main_process:
        logger.info(f"开始评估 [{split}]，共 {len(loader.dataset)} 样本…")

    for step, batch in enumerate(loader):
        rgb        = batch["rgb"]         # 已在 GPU（accelerate 处理）
        enc_target = batch["enc_target"]  # [B, T]

        outputs = model(image=rgb, text=None)

        # 取 encoder 全局预测（index=0）
        if isinstance(outputs, dict):
            pred_logits = torch.zeros(rgb.shape[0], num_classes, device=rgb.device)
        elif outputs.dim() == 3:
            pred_logits = outputs[:, 0, :]   # [B, num_classes]
        else:
            pred_logits = outputs

        pred_probs = torch.softmax(pred_logits, dim=-1)  # [B, num_classes]

        last_label = enc_target[:, -1].clamp(0, num_classes - 1)
        one_hot    = F.one_hot(last_label, num_classes=num_classes).float()

        # gather_for_metrics：跨 GPU 收集，自动去除 DataLoader padding 引入的重复数据
        pred_probs, one_hot = accelerator.gather_for_metrics((pred_probs, one_hot))
        all_probs.append(pred_probs.cpu())
        all_labels.append(one_hot.cpu())

        if accelerator.is_main_process and step % 50 == 0:
            mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            logger.info(f"  [{step}/{len(loader)}] mem={mem_mb:.0f}MB")

    probs_mat  = torch.cat(all_probs,  dim=0).numpy().T   # [C, N]
    labels_mat = torch.cat(all_labels, dim=0).numpy().T   # [C, N]

    # ── 指标计算 ──────────────────────────────────────────────────
    map_result = frame_level_map(probs_mat, labels_mat, with_bg=with_bg)
    mAP = map_result["map"] * 100
    cAP = map_result["cap"] * 100

    pred_cls  = np.argmax(probs_mat, axis=0)
    label_cls = np.argmax(labels_mat, axis=0)
    f1_result = compute_f1_per_class(pred_cls, label_cls, num_classes=num_classes)
    macro_f1  = f1_result["macro_f1"] * 100

    # ── 日志（仅主进程）──────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info(f"[{split}] 样本数: {probs_mat.shape[1]}")
        logger.info(f"  mAP    = {mAP:.2f}%")
        logger.info(f"  cAP    = {cAP:.2f}%")
        logger.info(f"  F1     = {macro_f1:.2f}%")
        logger.info("-" * 60)
        start_idx = 0 if with_bg else 1
        for i, (ap, acp) in enumerate(
            zip(map_result["all_cls_ap"], map_result["all_cls_acp"])
        ):
            ci = start_idx + i
            cn = class_names[ci] if ci < len(class_names) else f"cls_{ci}"
            f1 = float(f1_result["per_class_f1"][i]) * 100 if i < len(f1_result["per_class_f1"]) else 0.0
            logger.info(f"  {cn:30s} AP={ap*100:6.2f}%  cAP={acp*100:6.2f}%  F1={f1:6.2f}%")
        logger.info("=" * 60)

    return {
        "probs":          probs_mat,
        "labels":         labels_mat,
        "preds":          pred_cls,
        "map":            float(mAP),
        "cap":            float(cAP),
        "macro_f1":       float(macro_f1),
        "per_class_ap":   map_result["all_cls_ap"],
        "per_class_acp":  map_result["all_cls_acp"],
        "per_class_f1":   f1_result["per_class_f1"],
        "per_class_acc":  f1_result["per_class_acc"],
        "class_names":    class_names,
    }


# ────────────────────────────────────────────────────────────────────
# 结果保存
# ────────────────────────────────────────────────────────────────────

def save_results(results: dict[str, Any], output_dir: Path, split: str) -> None:
    """将预测矩阵与指标序列化保存。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 原始预测矩阵（供后续分析）
    pred_file = output_dir / f"predictions_{split}_{ts}.npz"
    np.savez_compressed(
        pred_file,
        probs=results["probs"],
        labels=results["labels"],
        preds=results["preds"],
    )
    logger.info(f"预测结果已保存: {pred_file}")

    # 结构化指标 JSON
    metrics = {
        "split":     split,
        "timestamp": ts,
        "mAP":       results["map"],
        "cAP":       results["cap"],
        "macro_F1":  results["macro_f1"],
        "per_class_ap": {
            name: float(ap)
            for name, ap in zip(results["class_names"][1:], results["per_class_ap"])
        },
        "per_class_acp": {
            name: float(v)
            for name, v in zip(results["class_names"][1:], results["per_class_acp"])
        },
        "per_class_f1": {
            name: float(v)
            for name, v in zip(results["class_names"][1:], results["per_class_f1"])
        },
    }
    metrics_file = output_dir / f"metrics_{split}_{ts}.json"
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"指标已保存: {metrics_file}")


# ────────────────────────────────────────────────────────────────────
# 主函数
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Accelerator 初始化 ────────────────────────────────────────
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed)

    # ── 配置加载 ──────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # CLI 覆盖
    if args.data_dir:
        cfg.setdefault("data", {})["dir"] = args.data_dir
    if args.metadata_csv:
        cfg.setdefault("data", {})["metadata_csv"] = args.metadata_csv
    if args.batch_size:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size

    # ── 输出目录 ──────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent / "eval_results"

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, rank=0 if accelerator.is_main_process else 1)

    if accelerator.is_main_process:
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"输出目录:   {output_dir}")
        logger.info(f"进程数:     {accelerator.num_processes}")

    # ── 数据加载器 ────────────────────────────────────────────────
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    batch_size = cfg.get("train", {}).get("batch_size", 64)

    _, val_loader = build_dataloaders(
        data_dir=data_cfg["dir"],
        metadata_csv=data_cfg["metadata_csv"],
        enc_steps=model_cfg["enc_steps"],
        dec_steps=model_cfg.get("dec_steps", 8),
        batch_size=batch_size,
        val_ratio=data_cfg.get("val_ratio", 0.1),
        nonzero_threshold=0,
        stride=model_cfg["enc_steps"],   # 不重叠，全覆盖
        num_workers=args.num_workers,
        pin_memory=True,
        preload=data_cfg.get("preload", True),
        num_preload_workers=data_cfg.get("num_preload_workers", 8),
        bg_weight=0.1,
        use_weighted_sampler=False,
        distributed=False,
        seed=args.seed,
        drop_last=False,
    )

    class_names = val_loader.dataset.class_names
    num_classes  = len(class_names)

    if accelerator.is_main_process:
        logger.info(f"类别数={num_classes}  验证样本={len(val_loader.dataset)}")

    # ── 模型构建 ──────────────────────────────────────────────────
    import clip as clip_lib
    clip_model, _ = clip_lib.load(model_cfg.get("clip_backbone", "ViT-B/32"), device="cpu")

    model = build_model(
        clip_model=clip_model,
        num_class=num_classes,
        enc_steps=model_cfg["enc_steps"],
        decoder_query_frames=model_cfg.get("dec_steps", 8),
        encoder_layers=model_cfg.get("encoder_layers", 3),
        decoder_layers=model_cfg.get("decoder_layers", 5),
        read_from=model_cfg.get("read_from", "feat"),
        zero_shot=model_cfg.get("zero_shot", False),
        add_fuse=model_cfg.get("add_fuse", False),
        freeze_mode="both",  # 评估时不冻结，需要完整前向
    )

    # ── accelerate.prepare：自动处理设备迁移与分布式包装 ──────────
    model, val_loader = accelerator.prepare(model, val_loader)

    # ── 加载 Checkpoint ───────────────────────────────────────────
    # accelerate.load_state 恢复模型权重（unwrap DDP 后加载）
    # 评估时不需要恢复优化器/调度器/RNG 状态，load_state 默认加载模型权重
    accelerator.load_state(args.checkpoint)
    if accelerator.is_main_process:
        logger.info(f"模型权重已从 {args.checkpoint} 加载")

    # ── 评估 ──────────────────────────────────────────────────────
    results = run_evaluation(
        accelerator=accelerator,
        model=model,
        loader=val_loader,
        class_names=class_names,
        with_bg=args.with_bg,
        split=args.split,
    )

    # ── 保存结果（仅主进程）──────────────────────────────────────
    if accelerator.is_main_process and args.save_preds and results:
        save_results(results, output_dir, split=args.split)
        logger.info(
            f"\n评估汇总:\n"
            f"  mAP = {results['map']:.2f}%\n"
            f"  cAP = {results['cap']:.2f}%\n"
            f"  F1  = {results['macro_f1']:.2f}%"
        )


if __name__ == "__main__":
    main()
