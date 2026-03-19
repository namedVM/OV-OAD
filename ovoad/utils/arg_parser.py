"""
两阶段参数解析公共模块
=====================

设计目标
--------
为 ``train.py`` 和 ``evaluate.py`` 提供统一的两阶段参数解析机制：

**第一阶段**
    仅解析 ``--config`` 路径，加载 YAML 配置文件，获得基准默认值。

**第二阶段**
    遍历已加载 config 的所有叶节点，动态构建完整的 ``ArgumentParser``。
    CLI 参数命名格式为 ``--k1__k2``（双下划线分隔嵌套层级），
    双下划线设计是为了兼容参数名本身含单下划线的情况，避免歧义。

优先级（从高到低）
------------------
1. CLI 显式传入的值
2. config 文件中的值（作为参数默认值）
3. 代码中的硬编码默认值（config 不含该键时回退）

写回机制
--------
两阶段解析完成后，所有最终生效的参数值（无论来源）都按照
``k1__k2`` → ``cfg['k1']['k2']`` 的映射关系写回 cfg 字典，
确保后续序列化保存时不丢失任何配置。

用法示例
--------
.. code-block:: python

    from utils.arg_parser import build_two_stage_parser, writeback_args_to_cfg

    # 可选：定义脚本专属的额外参数（不属于 config 嵌套结构的顶层参数）
    def add_extra_args(parser):
        parser.add_argument("--seed",           type=int,   default=42)
        parser.add_argument("--mixed-precision", type=str,  default="fp16",
                            choices=["no", "fp16", "bf16"])
        parser.add_argument("--debug",          action="store_true")
        parser.add_argument("--tag",            type=str,   default="")

    cfg, args = build_two_stage_parser(
        description="OV-OAD 分布式训练",
        extra_args_fn=add_extra_args,
    )
    # cfg 已写回所有 CLI 覆盖值，args 保留所有解析后的 Namespace 字段
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

# ────────────────────────────────────────────────────────────────────
# 内部辅助函数
# ────────────────────────────────────────────────────────────────────


def _load_yaml(path: str) -> dict[str, Any]:
    """从磁盘加载 YAML 配置文件，返回字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten_cfg(
    cfg: dict[str, Any],
    prefix: str = "",
    sep: str = "__",
) -> dict[str, Any]:
    """将嵌套 config 字典递归展平为 ``key1__key2 → value`` 的平铺字典。

    仅叶节点（非 dict 的值）会被纳入，中间节点（子 dict）不产生条目。

    Args:
        cfg:    待展平的字典
        prefix: 递归时的键前缀（外部调用无需传）
        sep:    层级分隔符，默认 ``__``（双下划线）

    Returns:
        平铺后的字典 ``{flat_key: leaf_value, ...}``
    Example::

        >>> _flatten_cfg({"train": {"lr": 1e-4, "epochs": 30}})
        {"train__lr": 1e-4, "train__epochs": 30}
    """
    result: dict[str, Any] = {}
    for key, value in cfg.items():
        flat_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            # 递归展平子字典
            result.update(_flatten_cfg(value, prefix=flat_key, sep=sep))
        else:
            result[flat_key] = value
    return result


def _infer_type(value: Any) -> type:
    """根据 config 中的值类型推断 argparse 参数类型。

    - ``bool``  → 不设置 type（由 ``action="store_true/false"`` 处理）
    - ``list``  → ``str``（CLI 传入后需再转换；目前作为字符串保留）
    - 其余      → 原类型（int / float / str）
    """
    if isinstance(value, bool):
        return bool  # 占位，实际由调用处特殊处理
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    if isinstance(value, list):
        return str  # list 暂以字符串接收，保持 config 原值
    return str


def _add_cfg_argument(
    parser: argparse.ArgumentParser,
    flat_key: str,
    default_value: Any,
) -> None:
    """向 parser 动态添加一个来自 config 的参数。

    参数命名规则：``flat_key`` 中的下划线不替换，直接拼接 ``--``。
    例如 ``train__lr`` → ``--train__lr``，``train__weight_decay`` → ``--train__weight_decay``。

    对 bool 类型特殊处理：使用 ``BooleanOptionalAction``（Python 3.9+）
    或手动提供 ``store_true`` / ``store_false`` 两个参数。

    Args:
        parser:        目标 ArgumentParser
        flat_key:      展平后的键，如 ``train__lr``
        default_value: 该键在 config 中的当前值（用作 argparse default）
    """
    arg_name = f"--{flat_key}"

    if isinstance(default_value, bool):
        # bool 参数使用 BooleanOptionalAction：支持 --flag / --no-flag
        parser.add_argument(
            arg_name,
            default=default_value,
            action=argparse.BooleanOptionalAction,
            help=f"Config 字段 {flat_key!r}，默认: {default_value}",
        )
    elif isinstance(default_value, list):
        # list 类型：接受多个值（nargs='+'），元素类型取第一个元素类型推断
        elem_type = type(default_value[0]) if default_value else str
        parser.add_argument(
            arg_name,
            nargs="+",
            type=elem_type,
            default=default_value,
            help=f"Config 字段 {flat_key!r}，默认: {default_value}",
        )
    else:
        value_type = _infer_type(default_value)
        parser.add_argument(
            arg_name,
            type=value_type,
            default=default_value,
            help=f"Config 字段 {flat_key!r}，默认: {default_value}",
        )


# ────────────────────────────────────────────────────────────────────
# 写回机制
# ────────────────────────────────────────────────────────────────────


def writeback_args_to_cfg(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    sep: str = "__",
    flat_keys: Optional[set[str]] = None,
) -> dict[str, Any]:
    """将 args 中所有以 ``sep`` 分隔的键值写回 cfg 字典对应的嵌套位置。

    仅处理 ``flat_keys`` 中列出的键（即来源于 config 展平的那些），
    跳过 ``config``、``seed`` 等脚本专属参数，避免污染 cfg 结构。

    写回逻辑：
        ``args.train__lr = 0.001``  →  ``cfg["train"]["lr"] = 0.001``

    Args:
        args:      argparse 解析后的 Namespace
        cfg:       原始 config 字典（in-place 修改后返回）
        sep:       层级分隔符，默认 ``__``
        flat_keys: 允许写回的键集合（通常为 _flatten_cfg 产生的 key 集合）

    Returns:
        写回后的 cfg 字典（同一对象，已 in-place 修改）
    """
    ns_dict = vars(args)
    for attr_name, value in ns_dict.items():
        # attr_name 是 argparse 将 "--train__lr" 转换后的属性名
        # argparse 会将 "-" 替换为 "_"，但我们用 "__" 分隔，所以需还原
        # 实际上 argparse 会将 "--train__lr" 映射为 args.train__lr（双下划线保留）
        flat_key = attr_name  # argparse 对 "--" 之后的部分，"-" → "_"

        # 仅处理来自 config 展平的键
        if flat_keys is not None and flat_key not in flat_keys:
            continue

        # 按 sep 拆分，写回嵌套字典
        parts = flat_key.split(sep)
        if len(parts) < 2:
            # 顶层键（非嵌套）直接写回
            cfg[flat_key] = value
            continue

        # 递归定位并写入
        node = cfg
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    return cfg


# ────────────────────────────────────────────────────────────────────
# 主入口：两阶段解析
# ────────────────────────────────────────────────────────────────────


def build_two_stage_parser(
    description: str = "",
    extra_args_fn: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    default_config: str = "configs/train.yml",
    argv: Optional[list[str]] = None,
) -> tuple[dict[str, Any], argparse.Namespace]:
    """执行两阶段参数解析，返回 (cfg, args)。

    **第一阶段**：仅解析 ``--config`` 参数，加载 YAML 文件作为默认值基准。

    **第二阶段**：基于展平后的 config 动态注册所有 ``--k1__k2`` 参数，
    再调用 ``extra_args_fn`` 注册脚本专属参数（如 ``--seed``、
    ``--checkpoint`` 等），最后完整解析命令行。

    **写回**：将所有最终生效的 config 参数值写回 cfg 字典。

    Args:
        description:    ArgumentParser description 字符串
        extra_args_fn:  可选回调，签名为 ``(parser) -> None``，
                        用于向 parser 添加脚本专属参数（不属于 config 嵌套结构）。
        default_config: ``--config`` 参数的硬编码默认路径。
        argv:           显式传入的命令行参数列表（默认 None 使用 sys.argv）。

    Returns:
        ``(cfg, args)`` 元组：
        - ``cfg``：写回所有最终参数值后的完整配置字典
        - ``args``：argparse.Namespace，包含所有解析后的参数

    Example::

        cfg, args = build_two_stage_parser(
            description="OV-OAD 训练",
            extra_args_fn=lambda p: (
                p.add_argument("--seed", type=int, default=42),
                p.add_argument("--eval-only", action="store_true"),
            ),
            default_config="configs/train.yml",
        )
    """
    # ── 第一阶段：仅解析 --config ──────────────────────────────────
    # 使用 parse_known_args 避免因未知参数报错
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="YAML 配置文件路径（第一阶段仅解析此参数）",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args(argv)

    # 加载 config 文件，获得基准默认值
    config_path = pre_args.config
    cfg = _load_yaml(config_path)

    # ── 第二阶段：动态构建完整 ArgumentParser ─────────────────────
    parser = argparse.ArgumentParser(
        description=description,
        # 允许 --config 同时出现在第二阶段（不报 unrecognized argument）
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 2a. 注册 --config（保留在最终 args 中，方便后续记录）
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="YAML 配置文件路径",
    )

    # 2b. 展平 config，动态注册所有 --k1__k2 参数
    #     以 config 值作为 default，实现"config 值优先于代码默认值"
    flat_cfg = _flatten_cfg(cfg)
    for flat_key, default_value in flat_cfg.items():
        _add_cfg_argument(parser, flat_key, default_value)

    # 2c. 注册脚本专属的额外参数（seed、checkpoint、eval-only 等）
    if extra_args_fn is not None:
        extra_args_fn(parser)

    # 2d. 完整解析命令行（包含 remaining_argv 与原始 argv 中的 --config）
    args = parser.parse_args(argv)

    # ── 写回：将所有最终生效的 config 参数写回 cfg ────────────────
    # 只写回来自 config 展平的那些键，跳过脚本专属参数
    cfg = writeback_args_to_cfg(args, cfg, flat_keys=set(flat_cfg.keys()))

    return cfg, args
