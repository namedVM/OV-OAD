"""
通用工具：日志配置、进度统计。

分布式初始化职责已完全转移给 accelerate.Accelerator，
本模块仅保留日志设置和统计工具。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any


def setup_logging(
    log_dir: str | Path,
    rank: int = 0,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """配置结构化日志，同时输出到终端和文件。

    仅主进程（rank=0）写入日志文件；所有进程均输出到终端。
    与 accelerate 日志系统兼容（不冲突）。

    Args:
        log_dir:   日志文件目录
        rank:      当前进程 rank（由 accelerate.is_main_process 决定传 0 或 1）
        log_level: 日志级别

    Returns:
        根 Logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 终端输出（所有进程）
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(log_level)
    root_logger.addHandler(ch)

    # 文件输出（仅主进程）
    if rank == 0:
        fh = logging.FileHandler(log_dir / "train.log", mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(log_level)
        root_logger.addHandler(fh)

    return root_logger


class AverageMeter:
    """跟踪指标的均值与最近值，兼容 timm.utils.AverageMeter 接口。"""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __repr__(self) -> str:
        return f"{self.name}: val={self.val:.4f} avg={self.avg:.4f}"
