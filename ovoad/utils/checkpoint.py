"""
Checkpoint 工具（基于 Hugging Face Accelerate）

Accelerate 已完整覆盖所有 checkpoint 需求：
  - save_state()  ：保存模型权重、优化器、调度器、RNG 状态
  - load_state()  ：全量恢复
  - ProjectConfiguration(total_limit=N)：自动管理 checkpoint 数量上限

本模块仅保留两个轻量级辅助函数：
  1. save_training_state  — 在 accelerate checkpoint 目录旁写入 training_state.json
     （记录 epoch / step / best_metrics 等轻量元数据，accelerate 本身不持久化这些）
  2. load_training_state  — 读取上述 JSON，供续训时获取 epoch / best_metrics
  3. find_latest_checkpoint — 在输出目录中查找最新的 accelerate checkpoint 路径

不再手动编写：
  × DDP 模型解包
  × RNG 状态序列化（random / numpy / torch / cuda）
  × AMP GradScaler 状态保存
  × 优化器 / 调度器 state_dict 管理
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def save_training_state(
    ckpt_dir: Path,
    epoch: int,
    step: int,
    best_metrics: dict[str, float],
) -> None:
    """在 accelerate checkpoint 目录中写入训练进度元数据。

    accelerate.save_state 自身不保存 epoch、step、best_metrics，
    通过此函数补充写入 training_state.json。

    Args:
        ckpt_dir:     accelerate.save_state 输出的目录（如 checkpoints/step_0010000）
        epoch:        当前 epoch（从 0 开始）
        step:         当前全局 optimizer step
        best_metrics: 最优指标字典，如 {'map': 37.5, 'cap': 73.8, 'macro_f1': 0.6}
    """
    state = {
        "epoch":        epoch,
        "step":         step,
        "best_metrics": best_metrics,
    }
    state_file = Path(ckpt_dir) / "training_state.json"
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug(f"training_state.json 已写入: {state_file}")


def load_training_state(ckpt_dir: str | Path) -> dict[str, Any]:
    """从 accelerate checkpoint 目录读取训练进度元数据。

    Args:
        ckpt_dir: checkpoint 目录路径

    Returns:
        包含 epoch、step、best_metrics 的字典；文件不存在时返回默认值
    """
    state_file = Path(ckpt_dir) / "training_state.json"
    if not state_file.exists():
        logger.warning(f"training_state.json 不存在: {state_file}，使用默认值")
        return {"epoch": 0, "step": 0, "best_metrics": {}}

    state = json.loads(state_file.read_text(encoding="utf-8"))
    logger.info(
        f"已读取 training_state: epoch={state.get('epoch', 0)}, "
        f"step={state.get('step', 0)}, "
        f"best_metrics={state.get('best_metrics', {})}"
    )
    return state


def find_latest_checkpoint(ckpt_dir: str | Path) -> Optional[Path]:
    """在 checkpoint 目录中查找最新的 accelerate checkpoint。

    accelerate 的 ProjectConfiguration(automatic_checkpoint_naming=True) 默认使用
    step_<N> 格式命名，本函数按步数降序返回最新的目录。

    Args:
        ckpt_dir: checkpoint 根目录（如 run_xxx/checkpoints）

    Returns:
        最新 checkpoint 目录的 Path，若无则返回 None
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None

    # 优先：step_<N> 格式
    step_pattern = re.compile(r"^step_(\d+)$")
    step_dirs = [
        (int(m.group(1)), p)
        for p in ckpt_dir.iterdir()
        if p.is_dir() and (m := step_pattern.match(p.name))
    ]
    if step_dirs:
        step_dirs.sort(key=lambda x: x[0], reverse=True)
        latest = step_dirs[0][1]
        logger.info(f"自动续训：找到最新 checkpoint {latest}")
        return latest

    # 次选：epoch_<N> 格式
    epoch_pattern = re.compile(r"^epoch_(\d+)$")
    epoch_dirs = [
        (int(m.group(1)), p)
        for p in ckpt_dir.iterdir()
        if p.is_dir() and (m := epoch_pattern.match(p.name))
    ]
    if epoch_dirs:
        epoch_dirs.sort(key=lambda x: x[0], reverse=True)
        latest = epoch_dirs[0][1]
        logger.info(f"自动续训：找到最新 checkpoint {latest}")
        return latest

    logger.info(f"未在 {ckpt_dir} 找到 checkpoint，将从头开始训练")
    return None
