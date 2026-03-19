"""
OV-OAD 分布式训练脚本（基于 Hugging Face Accelerate）

核心依赖 `accelerate` 库统一处理：
  - 多节点多卡分布式（torchrun / DeepSpeed / FSDP）
  - 混合精度（fp16 / bf16）
  - 梯度累积
  - Checkpoint 保存与完整状态恢复（模型/优化器/调度器/RNG）
  - TensorBoard / WandB 日志追踪

用法（单机多卡）：
    torchrun --nproc_per_node=8 train.py --config configs/train.yml

用法（多节点，使用 accelerate launch）：
    accelerate launch --config_file accelerate_config.yml train.py \
        --config configs/train.yml

用法（eval-only）：
    python train.py --config configs/train.yml \
        --resume outputs/run_xxx/checkpoints/step_1000 \
        --eval-only

参数覆盖示例（两阶段解析，双下划线分隔嵌套键）：
    python train.py --config configs/train.yml \
        --train__lr 1e-4 --train__epochs 50 --model__enc_steps 64
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from ovoad.datasets import build_dataloaders
from ovoad.models import build_model
from ovoad.utils import (
    AverageMeter,
    compute_f1_per_class,
    find_latest_checkpoint,
    frame_level_map,
    load_training_state,
    save_training_state,
    setup_logging,
)
from ovoad.utils.arg_parser import build_two_stage_parser

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# 配置管理
# ────────────────────────────────────────────────────────────────────


def _add_train_extra_args(parser: argparse.ArgumentParser) -> None:
    """注册训练脚本专属的额外 CLI 参数（不属于 config 嵌套结构）。

    这些参数不会通过 ``k1__k2`` 双下划线机制写回 cfg，
    而是由 main() 直接读取 args.xxx 使用。

    注意：config 顶层已有 ``seed`` 和 ``debug`` 字段，
    两阶段解析会自动将其注册为 ``--seed`` 和 ``--debug``，
    因此这里只注册 config 中不存在的专属控制参数。
    """
    # Checkpoint / 续训控制
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从指定 checkpoint 目录恢复训练",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="自动从 output 目录中最新的 checkpoint 续训",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="仅运行评估，跳过训练循环",
    )

    # 混合精度（Accelerate 运行时选项，不属于 config 结构）
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="混合精度训练类型（fp16 / bf16 / no）",
    )

    # 实验标签（仅影响运行目录命名，不属于 config 结构）
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="实验标签，附加到运行目录名称后缀",
    )


def flatten_config(cfg, prefix=""):
    """将嵌套字典转为扁平字典，并将列表转为字符串"""
    items = {}
    for k, v in cfg.items():
        new_key = f"{prefix}{k}" if prefix == "" else f"{prefix}/{k}"
        if isinstance(v, dict):
            items.update(flatten_config(v, new_key))
        elif isinstance(v, (list, tuple)):
            items[new_key] = str(v)  # 将 [0.9, 0.999] 转为 "[0.9, 0.999]"
        elif v is None:
            items[new_key] = "None"
        else:
            items[new_key] = v
    return items


def make_run_dir(output_root: str | Path, tag: str = "") -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{ts}" + (f"_{tag}" if tag else "")
    run_dir = Path(output_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ────────────────────────────────────────────────────────────────────
# 优化器 & 调度器
# ────────────────────────────────────────────────────────────────────


def build_optimizer(
    model: torch.nn.Module,
    optimizer_type: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """构建优化器，bias / LayerNorm 参数不施加 weight_decay。"""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ("bias", "norm", "ln_")):
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    opt_lower = optimizer_type.lower()
    if opt_lower == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif opt_lower == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    elif opt_lower == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"未知 optimizer_type: {optimizer_type!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """构建学习率调度器。"""
    if scheduler_type == "cosine":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=total_steps,
        )
    elif scheduler_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[pg["lr"] for pg in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=warmup_steps / max(total_steps, 1),
        )
    else:
        raise ValueError(f"未知 scheduler_type: {scheduler_type!r}")


# ────────────────────────────────────────────────────────────────────
# 验证
# ────────────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: list[str],
    epoch: int,
    global_step: int,
) -> dict[str, float]:
    """分布式验证：accelerate 自动处理跨 GPU 的数据收集。"""
    model.eval()
    num_classes = len(class_names)

    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        rgb = batch["rgb"]  # [B, T, D]，accelerate 已移到正确设备
        enc_target = batch["enc_target"]  # [B, T]

        outputs = model(image=rgb, text=None)
        # 有监督推理输出：[B, 1+dec_q, num_classes]，取 index=0（encoder 全局预测）
        if isinstance(outputs, dict):
            pred_logits = torch.zeros(rgb.shape[0], num_classes, device=rgb.device)
        elif outputs.dim() == 3:
            pred_logits = outputs[:, 0, :]  # [B, num_classes]
        else:
            pred_logits = outputs

        pred_probs = torch.softmax(pred_logits, dim=-1)  # [B, num_classes]

        last_label = enc_target[:, -1].clamp(0, num_classes - 1)
        one_hot = F.one_hot(last_label, num_classes=num_classes).float()  # [B, C]

        # accelerate gather_for_metrics：自动处理 padding 并去除重复数据
        pred_probs, one_hot = accelerator.gather_for_metrics((pred_probs, one_hot))
        all_probs.append(pred_probs.cpu())
        all_labels.append(one_hot.cpu())

    probs_mat = torch.cat(all_probs, dim=0).numpy().T  # [C, N]
    labels_mat = torch.cat(all_labels, dim=0).numpy().T  # [C, N]

    map_result = frame_level_map(probs_mat, labels_mat, with_bg=False)
    mAP = map_result["map"] * 100
    cAP = map_result["cap"] * 100

    pred_cls = np.argmax(probs_mat, axis=0)
    label_cls = np.argmax(labels_mat, axis=0)
    f1_result = compute_f1_per_class(pred_cls, label_cls, num_classes=num_classes)
    macro_f1 = f1_result["macro_f1"] * 100

    if accelerator.is_main_process:
        ap_arr = map_result["all_cls_ap"]
        logger.info(
            f"[Epoch {epoch}] mAP={mAP:.2f}%  cAP={cAP:.2f}%  macro-F1={macro_f1:.2f}%"
        )
        for i, ap in enumerate(ap_arr):
            cls_name = (
                class_names[i + 1] if i + 1 < len(class_names) else f"cls_{i + 1}"
            )
            logger.info(f"  {cls_name}: AP={ap * 100:.2f}%")

        # accelerate 内置日志追踪（支持 TensorBoard / WandB）
        accelerator.log(
            {"val/mAP": mAP, "val/cAP": cAP, "val/macro_F1": macro_f1},
            step=global_step,
        )

    return {"map": mAP, "cap": cAP, "macro_f1": macro_f1}


# ────────────────────────────────────────────────────────────────────
# 主训练流程
# ────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── 两阶段参数解析 ────────────────────────────────────────────
    # 第一阶段：解析 --config，加载 YAML 作为默认值基准
    # 第二阶段：基于展平的 config 动态构建完整 ArgumentParser，
    #           注册所有 --k1__k2 参数（config 值为 default），
    #           再注册训练脚本专属参数（--resume / --seed / --eval-only 等）
    # 写回：将所有最终生效值按 k1__k2 → cfg[k1][k2] 写回 cfg 字典
    cfg, args = build_two_stage_parser(
        description="OV-OAD 分布式训练（Accelerate）",
        extra_args_fn=_add_train_extra_args,
        default_config="configs/train.yml",
    )

    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    ckpt_cfg = cfg.get("checkpoint", {})
    eval_cfg = cfg.get("evaluate", {})

    # ── 输出目录 ──────────────────────────────────────────────────
    output_root = cfg["output"]["dir"]
    if args.eval_only and args.resume:
        run_dir = Path(args.resume).parent.parent  # checkpoint → run_dir
    else:
        run_dir = make_run_dir(output_root, tag=args.tag)

    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Accelerator 初始化 ────────────────────────────────────────
    # ProjectConfiguration：自动管理 checkpoint 目录与最大保留数
    project_config = ProjectConfiguration(
        project_dir=str(run_dir),
        logging_dir=str(log_dir),
        automatic_checkpoint_naming=False,  # 自动命名：step_<N>
        total_limit=ckpt_cfg.get("keep", 5),  # 最多保留 N 个 checkpoint
    )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,  # fp16 / bf16 / no
        gradient_accumulation_steps=train_cfg.get("accum_steps", 1),
        log_with="tensorboard",  # 内置 TensorBoard 追踪
        project_config=project_config,
    )

    # ── 日志（仅主进程写文件）────────────────────────────────────
    setup_logging(log_dir, rank=0 if accelerator.is_main_process else 1)

    if accelerator.is_main_process:
        logger.info(f"运行目录: {run_dir}")
        logger.info(
            f"进程数: {accelerator.num_processes}  混合精度: {args.mixed_precision}"
        )
        # 保存配置快照
        with open(run_dir / "config.yml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        accelerator.init_trackers(
            project_name="ovoad",
            config=flatten_config(cfg),
            init_kwargs={"tensorboard": {"flush_secs": 30}},
        )

    # ── 随机种子（rank-aware）────────────────────────────────────
    set_seed(args.seed)  # accelerate 的 set_seed 自动对每个进程偏移种子

    # ── 数据加载器 ────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        data_dir=data_cfg["dir"],
        metadata_csv=data_cfg["metadata_csv"],
        enc_steps=model_cfg["enc_steps"],
        dec_steps=model_cfg.get("dec_steps", 8),
        batch_size=train_cfg["batch_size"],
        val_ratio=data_cfg.get("val_ratio", 0.1),
        nonzero_threshold=data_cfg.get("nonzero_threshold", 0),
        stride=data_cfg.get("stride", 1),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        preload=data_cfg.get("preload", True),
        num_preload_workers=data_cfg.get("num_preload_workers", 8),
        bg_weight=data_cfg.get("bg_weight", 0.1),
        # accelerate 已内置分布式 sampler，无需手动设置
        use_weighted_sampler=data_cfg.get("use_weighted_sampler", False),
        distributed=False,
        seed=args.seed,
        drop_last=True,
    )

    class_names = train_loader.dataset.class_names
    num_classes = len(class_names)

    if accelerator.is_main_process:
        logger.info(
            f"类别数={num_classes}  训练样本={len(train_loader.dataset)}"
            f"  验证样本={len(val_loader.dataset)}"
        )

    # ── 模型构建 ──────────────────────────────────────────────────
    import clip as clip_lib

    clip_model, _ = clip_lib.load(
        model_cfg.get("clip_backbone", "ViT-B/32"), device="cpu"
    )

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
        freeze_mode=model_cfg.get("freeze_mode", "none"),
        loss_weight_enc=model_cfg.get("loss_weight_enc", 1.0),
        loss_weight_dec=model_cfg.get("loss_weight_dec", 1.0),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        logger.info(f"可训练参数: {n_params / 1e6:.2f}M")

    # ── 优化器 & 调度器 ───────────────────────────────────────────
    optimizer = build_optimizer(
        model=model,
        optimizer_type=train_cfg.get("optimizer", "adamw"),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    total_steps = train_cfg["epochs"] * len(train_loader)
    warmup_steps = train_cfg.get("warmup_epochs", 2) * len(train_loader)
    min_lr_ratio = train_cfg.get("min_lr", 1e-6) / train_cfg["lr"]

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=train_cfg.get("scheduler", "cosine"),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    # ── accelerate.prepare：一行完成 DDP 包装 + 设备迁移 ──────────
    # accelerate 自动为 DataLoader 添加 DistributedSampler
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ── Checkpoint 恢复 ───────────────────────────────────────────
    # accelerate.load_state 自动恢复：模型权重、优化器状态、
    # 调度器状态、RNG 状态（Python/NumPy/PyTorch/CUDA）
    start_epoch = 0
    global_step = 0
    best_metrics = {"map": 0.0, "cap": 0.0, "macro_f1": 0.0}

    resume_path: Optional[Path] = None
    if args.resume:
        resume_path = Path(args.resume)
    elif args.auto_resume or ckpt_cfg.get("auto_resume", False):
        resume_path = find_latest_checkpoint(ckpt_dir)
        if resume_path and accelerator.is_main_process:
            logger.info(f"自动续训：找到 {resume_path}")

    if resume_path is not None and resume_path.exists():
        # accelerate 读取 training_state.json 确定 epoch/step/best_metrics
        accelerator.load_state(str(resume_path))
        state_file = resume_path / "training_state.json"
        if state_file.exists():
            import json

            state = json.loads(state_file.read_text())
            start_epoch = state.get("epoch", 0) + 1
            global_step = state.get("step", 0)
            best_metrics = state.get("best_metrics", best_metrics)
        if accelerator.is_main_process:
            logger.info(
                f"从 epoch={start_epoch} 恢复，best_mAP={best_metrics['map']:.2f}%"
            )

    # ── 仅评估模式 ────────────────────────────────────────────────
    if args.eval_only:
        val_metrics = validate(
            accelerator,
            model,
            val_loader,
            class_names,
            epoch=0,
            global_step=0,
        )
        if accelerator.is_main_process:
            logger.info(f"评估完成: {val_metrics}")
        accelerator.end_training()
        return

    # ── 训练循环 ──────────────────────────────────────────────────
    total_epochs = train_cfg["epochs"]
    eval_freq = eval_cfg.get("eval_freq", 5)
    save_freq = ckpt_cfg.get("save_freq", 5)
    clip_grad = train_cfg.get("clip_grad", 5.0)
    print_freq = cfg.get("print_freq", 50)
    debug = args.debug
    zero_shot = model_cfg.get("zero_shot", False)  # 提前确定，避免每 step 重复查询

    if accelerator.is_main_process:
        logger.info(f"开始训练：epoch {start_epoch} → {total_epochs - 1}")

    for epoch in range(start_epoch, total_epochs):
        model.train()

        # DistributedSampler 需要每 epoch 同步随机种子
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        loss_meter = AverageMeter("loss")
        enc_meter = AverageMeter("loss_enc")
        dec_meter = AverageMeter("loss_dec")

        for step, batch in enumerate(train_loader):
            # accelerate 的梯度累积上下文管理器：
            # 在累积步内自动跳过 all_reduce，最后一步才同步梯度
            with accelerator.accumulate(model):
                rgb = batch["rgb"]
                enc_target = batch["enc_target"]
                dec_target = batch["dec_target"]
                B = rgb.shape[0]

                # zero_shot=True：传入 CLIP token（由 dataset 按 class_id 索引）
                #                 text shape [B, 1+dec_steps, 77]
                # zero_shot=False：传入有监督标注 (enc_target, dec_target)
                if zero_shot:
                    text_input = batch["text"]  # [B, 1+dec_q, 77]
                else:
                    text_input = (enc_target, dec_target)

                losses = model(image=rgb, text=text_input)
                loss_enc = losses.get(
                    "loss_enc_vtc", torch.tensor(0.0, device=rgb.device)
                )
                loss_dec = losses.get(
                    "loss_dec_vtc", torch.tensor(0.0, device=rgb.device)
                )
                total_loss = loss_enc + loss_dec

                # accelerate 处理 AMP scaler，只需调用 backward
                accelerator.backward(total_loss)

                # 梯度裁剪（accelerate 内置 unscale 逻辑）
                if clip_grad > 0 and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        clip_grad,
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 仅在真实的优化步结束时更新统计（跳过累积中间步）
            if accelerator.sync_gradients:
                global_step += 1

                # 跨进程规约 loss（用于准确记录）
                loss_reduced = accelerator.reduce(total_loss, reduction="mean")
                enc_reduced = accelerator.reduce(loss_enc, reduction="mean")
                dec_reduced = accelerator.reduce(loss_dec, reduction="mean")

                loss_meter.update(loss_reduced.item(), B)
                enc_meter.update(enc_reduced.item(), B)
                dec_meter.update(dec_reduced.item(), B)

                if accelerator.is_main_process:
                    # accelerate 内置日志（同时写 TensorBoard）
                    accelerator.log(
                        {
                            "train/total_loss": loss_meter.avg,
                            "train/loss_enc": enc_meter.avg,
                            "train/loss_dec": dec_meter.avg,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "sys/gpu_mem_mb": torch.cuda.max_memory_allocated()
                            / 1024**2,
                        },
                        step=global_step,
                    )

                    if global_step % print_freq == 0:
                        logger.info(
                            f"[Ep {epoch}][Step {global_step}] "
                            f"loss={loss_meter.avg:.4f} "
                            f"(enc={enc_meter.avg:.4f}, dec={dec_meter.avg:.4f}) "
                            f"lr={optimizer.param_groups[0]['lr']:.2e}"
                        )

            if debug and step >= 10:
                break

        if accelerator.is_main_process:
            logger.info(
                f"[Epoch {epoch}] 训练完成  "
                f"loss={loss_meter.avg:.4f} (enc={enc_meter.avg:.4f}, dec={dec_meter.avg:.4f})"
            )

        # ── 定期评估 ────────────────────────────────────────────
        is_last = epoch == total_epochs - 1
        if epoch % eval_freq == 0 or is_last:
            val_metrics = validate(
                accelerator,
                model,
                val_loader,
                class_names,
                epoch=epoch,
                global_step=global_step,
            )
            is_best = val_metrics["map"] > best_metrics["map"]
            if is_best:
                best_metrics = val_metrics

        # ── Checkpoint 保存 ──────────────────────────────────────
        # accelerate.save_state 自动保存：
        #   - 模型权重（unwrap 后的纯权重）
        #   - 优化器状态
        #   - 调度器状态
        #   - Python / NumPy / PyTorch / CUDA RNG 状态
        #   - 自定义注册对象（通过 register_for_checkpointing）
        if epoch % save_freq == 0 or is_last:
            ckpt_save_dir = ckpt_dir / f"step_{global_step:07d}"
            ckpt_save_dir.mkdir(parents=True, exist_ok=True)
            accelerator.save_state(str(ckpt_save_dir))

            # 额外保存训练进度（epoch、step、best_metrics），供续训使用
            if accelerator.is_main_process:
                import json

                state = {
                    "epoch": epoch,
                    "step": global_step,
                    "best_metrics": best_metrics,
                }
                (ckpt_save_dir / "training_state.json").write_text(
                    json.dumps(state, indent=2, ensure_ascii=False)
                )
                logger.info(f"Checkpoint 已保存: {ckpt_save_dir}")

                # 若为最优模型，额外硬链接一份 best_map
                best_dir = ckpt_dir / "best_map"
                if is_best and epoch % eval_freq == 0:
                    best_dir.parent.mkdir(parents=True, exist_ok=True)
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    shutil.copytree(ckpt_save_dir, best_dir)
                    logger.info(f"最优 mAP={best_metrics['map']:.2f}% → {best_dir}")

        accelerator.wait_for_everyone()  # 确保所有进程同步后再进入下一 epoch

    # ── 训练结束 ──────────────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info(f"训练完成！最优 mAP={best_metrics['map']:.2f}%")

    accelerator.end_training()  # flush 日志追踪器，关闭 TensorBoard writer


if __name__ == "__main__":
    main()
