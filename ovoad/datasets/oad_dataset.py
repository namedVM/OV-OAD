"""
OAD 数据集实现：加载由 .pth 文件 + metadata.csv 组成的在线动作检测数据集。

数据格式：
    - metadata.csv: 包含 'action'（动作描述）和 'id'（类别编号）两列，id=0 为背景
    - {sample_id}.pth: {'rgb': Tensor[T, D], 'anno': Tensor[T]}
        * rgb: CLIP 提取的逐帧特征，形状 [T, D]
        * anno: 逐帧动作类别标注（对应 metadata 的 id 列）

性能优化：
    - 所有 .pth 文件预加载到内存（可配置为按需加载）
    - 支持多进程预加载（prefetch）
    - 动态 padding + mask，保证 batch 内序列长度一致
    - 背景帧降权：class_weight 中背景类权重设为可配置的较小值
    - 支持 DistributedSampler
"""

from __future__ import annotations

import csv
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler


# ────────────────────────────────────────────────────────────────────
# 元数据读取
# ────────────────────────────────────────────────────────────────────

def load_metadata(
    csv_path: str | Path,
) -> tuple[dict[int, str], list[str], list[str]]:
    """读取 metadata.csv，返回 id→action 映射、有序类别名列表、有序文本描述列表。

    CSV 须包含 'id' 和 'action' 两列；若存在 'text' 列则优先用其作为
    CLIP 文本描述（支持更丰富的自然语言 prompt），否则回退到 'action'。

    Args:
        csv_path: metadata.csv 路径

    Returns:
        id2action:   {id: action_name}，id=0 为背景
        class_names: 按 id 排序的类别名列表（index 即 id）
        class_texts: 按 id 排序的 CLIP 文本描述列表（用于 tokenize）
    """
    id2action: dict[int, str] = {}
    id2text:   dict[int, str] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_text_col = "text" in fieldnames
        for row in reader:
            action_id   = int(row["id"])
            action_name = row["action"].strip()
            id2action[action_id] = action_name
            # "text" 列优先，不存在则回退到 "action"
            id2text[action_id] = row["text"].strip() if has_text_col else action_name

    # 按 id 排序确保 class_names[i] / class_texts[i] 对应 id=i
    max_id = max(id2action.keys())
    class_names: list[str] = []
    class_texts: list[str] = []
    for i in range(max_id + 1):
        class_names.append(id2action.get(i, f"class_{i}"))
        class_texts.append(id2text.get(i, id2action.get(i, f"class_{i}")))

    return id2action, class_names, class_texts


# ────────────────────────────────────────────────────────────────────
# 数据集主类
# ────────────────────────────────────────────────────────────────────

class OadFeatureDataset(Dataset):
    """在线动作检测特征数据集。

    从 .pth 文件加载 CLIP 预提取特征，并生成适配 OadTransformer 的
    滑动窗口样本（历史帧窗口 + 解码器预测帧）。

    Args:
        data_dir: 存放所有 .pth 文件的目录
        metadata_csv: metadata.csv 路径
        enc_steps: Encoder 历史帧窗口大小（OadTransformer num_tokens）
        dec_steps: Decoder 预测帧数
        split: "train" 或 "val"
        val_ratio: 验证集比例（按视频文件数量切分），默认 0.1
        nonzero_threshold: 窗口内非背景帧数量最低阈值（过滤无动作窗口）
        stride: 滑动窗口步长，默认 1
        preload: 是否预加载所有 .pth 到内存（推荐开启以消除 I/O 瓶颈）
        num_preload_workers: 预加载使用的线程数
        bg_weight: 背景类在类权重计算中的权重系数（< 1 实现降权）
        cache_dir: 可选的特征缓存目录（None 则不使用缓存）
        seed: 随机种子（用于 train/val 划分）
    """

    def __init__(
        self,
        data_dir: str | Path,
        metadata_csv: str | Path,
        enc_steps: int = 32,
        dec_steps: int = 8,
        split: str = "train",
        val_ratio: float = 0.1,
        nonzero_threshold: int = 0,
        stride: int = 1,
        preload: bool = True,
        num_preload_workers: int = 8,
        bg_weight: float = 0.1,
        cache_dir: Optional[str | Path] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        assert split in ("train", "val"), f"split 必须为 'train' 或 'val'，得到 {split!r}"

        self.data_dir = Path(data_dir)
        self.metadata_csv = Path(metadata_csv)
        self.enc_steps = enc_steps
        self.dec_steps = dec_steps
        self.split = split
        self.nonzero_threshold = nonzero_threshold
        self.stride = stride
        self.bg_weight = bg_weight

        # 读取类别信息（含 text 描述）
        self.id2action, self.class_names, class_texts = load_metadata(self.metadata_csv)
        self.num_classes = len(self.class_names)
        self.bg_class_id = 0  # id=0 保留为背景

        # 预 tokenize 所有类别文本描述，shape [num_classes, context_length(77)]
        # 延迟导入避免循环依赖；tokenize 返回 LongTensor
        import clip as _clip
        self.text_tokens: torch.Tensor = _clip.tokenize(class_texts)  # [C, 77]

        # 扫描所有 .pth 文件并按 train/val 划分
        all_pth_files = sorted(self.data_dir.glob("*.pth"))
        if len(all_pth_files) == 0:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到任何 .pth 文件")

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_pth_files))
        n_val = max(1, int(len(all_pth_files) * val_ratio))
        val_indices = set(perm[:n_val].tolist())

        if split == "train":
            self.pth_files = [all_pth_files[i] for i in range(len(all_pth_files)) if i not in val_indices]
        else:
            self.pth_files = [all_pth_files[i] for i in val_indices]

        # 预加载 / 按需加载
        self._features: dict[str, torch.Tensor] = {}   # {stem: rgb_tensor}
        self._annotations: dict[str, torch.Tensor] = {}  # {stem: anno_tensor}
        self._lock = threading.Lock()

        if preload:
            self._preload_all(num_workers=num_preload_workers)

        # 构建滑动窗口样本列表
        self.samples: list[tuple[str, int, int]] = []  # (stem, start, end)
        self._build_sample_index()

        # 计算类别权重（用于 WeightedRandomSampler 或损失加权）
        self.class_weights = self._compute_class_weights()

    # ──────────────────────────────────────────────────────────────
    # 数据预加载
    # ──────────────────────────────────────────────────────────────

    def _load_single(self, pth_file: Path) -> tuple[str, torch.Tensor, torch.Tensor]:
        """加载单个 .pth 文件。"""
        data = torch.load(pth_file, map_location="cpu", weights_only=True)
        rgb: torch.Tensor = data["rgb"].float()    # [T, D]
        anno: torch.Tensor = data["anno"].long()   # [T]
        return pth_file.stem, rgb, anno

    def _preload_all(self, num_workers: int = 8) -> None:
        """多线程并发预加载所有 .pth 文件到内存。"""
        print(f"[OadDataset] 预加载 {len(self.pth_files)} 个 .pth 文件 (workers={num_workers})…")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._load_single, f): f for f in self.pth_files}
            loaded = 0
            for future in as_completed(futures):
                stem, rgb, anno = future.result()
                self._features[stem] = rgb
                self._annotations[stem] = anno
                loaded += 1
                if loaded % 100 == 0:
                    print(f"  已加载 {loaded}/{len(self.pth_files)}")
        print(f"[OadDataset] 预加载完成，共 {len(self._features)} 个视频。")

    def _get_or_load(self, stem: str, pth_file: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """按需懒加载（preload=False 时使用）。"""
        if stem not in self._features:
            with self._lock:
                # 双重检查
                if stem not in self._features:
                    _, rgb, anno = self._load_single(pth_file)
                    self._features[stem] = rgb
                    self._annotations[stem] = anno
        return self._features[stem], self._annotations[stem]

    # ──────────────────────────────────────────────────────────────
    # 样本索引构建
    # ──────────────────────────────────────────────────────────────

    def _build_sample_index(self) -> None:
        """构建滑动窗口样本列表，过滤无动作窗口。"""
        stem2file = {f.stem: f for f in self.pth_files}
        total_frames = 0
        skipped = 0

        for stem, pth_file in stem2file.items():
            if stem in self._features:
                anno = self._annotations[stem]
            else:
                # 懒加载模式：读取 anno 检查是否加载
                _, anno = self._get_or_load(stem, pth_file)

            T = anno.shape[0]
            total_frames += T
            min_len = self.enc_steps + self.dec_steps

            if T < min_len:
                skipped += 1
                continue

            # 滑动窗口：[start, end) 为 enc 窗口，[end, end+dec_steps) 为 dec 窗口
            for start in range(0, T - min_len + 1, self.stride):
                end = start + self.enc_steps
                if end + self.dec_steps > T:
                    break

                # 过滤：enc 窗口内非背景帧数量低于阈值则跳过
                enc_anno = anno[start:end]
                n_nonzero = (enc_anno != self.bg_class_id).sum().item()
                if n_nonzero < self.nonzero_threshold:
                    skipped += 1
                    continue

                self.samples.append((stem, start, end))

        print(
            f"[OadDataset:{self.split}] 共 {len(stem2file)} 个视频，"
            f"{total_frames} 帧，构建 {len(self.samples)} 个样本（跳过 {skipped}）"
        )

    # ──────────────────────────────────────────────────────────────
    # 类别权重
    # ──────────────────────────────────────────────────────────────

    def _compute_class_weights(self) -> torch.Tensor:
        """统计各类别帧频率，计算逆频率权重（背景类额外降权）。

        Returns:
            class_weights: [num_classes]，背景类权重乘以 bg_weight
        """
        counts = torch.zeros(self.num_classes, dtype=torch.float)
        stem2file = {f.stem: f for f in self.pth_files}

        for stem, pth_file in stem2file.items():
            anno = self._annotations.get(stem)
            if anno is None:
                _, anno = self._get_or_load(stem, pth_file)
            for cls_id in range(self.num_classes):
                counts[cls_id] += (anno == cls_id).sum().float()

        # 逆频率：稀有类获得更高权重
        total = counts.sum()
        weights = total / (counts.clamp(min=1) * self.num_classes)

        # 背景类降权
        weights[self.bg_class_id] *= self.bg_weight

        # 归一化
        weights = weights / weights.sum() * self.num_classes
        return weights

    def get_sample_weights(self) -> torch.Tensor:
        """返回每个样本的采样权重（用于 WeightedRandomSampler）。

        每个样本的权重 = enc 窗口最后一帧的类别权重，
        使动作帧被更频繁采样。

        Returns:
            sample_weights: [len(self.samples)]
        """
        weights = []
        stem2file = {f.stem: f for f in self.pth_files}
        for stem, start, end in self.samples:
            anno = self._annotations.get(stem)
            if anno is None:
                _, anno = self._get_or_load(stem, stem2file[stem])
            last_frame_cls = anno[end - 1].item()
            weights.append(self.class_weights[last_frame_cls].item())
        return torch.tensor(weights, dtype=torch.float)

    # ──────────────────────────────────────────────────────────────
    # Dataset 接口
    # ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """返回单个样本。

        Returns:
            dict with:
                'rgb':        [enc_steps, D]          — encoder 输入特征
                'enc_target': [enc_steps]              — encoder 帧级类别标注
                'dec_target': [dec_steps]              — decoder 预测帧标注
                'mask':       [enc_steps]              — 有效帧掩码（全 1）
                'text':       [1+dec_steps, 77]        — CLIP token：
                                                          index 0   → encoder 当前帧（最后帧）文本
                                                          index 1:  → decoder 各预测帧文本
        """
        stem, start, end = self.samples[index]

        # 获取特征与标注
        rgb = self._features.get(stem)
        anno = self._annotations.get(stem)
        if rgb is None or anno is None:
            stem2file = {f.stem: f for f in self.pth_files}
            rgb, anno = self._get_or_load(stem, stem2file[stem])

        enc_rgb  = rgb[start:end].clone()                    # [enc_steps, D]
        enc_anno = anno[start:end].clone()                   # [enc_steps]
        dec_anno = anno[end: end + self.dec_steps].clone()   # [dec_steps]

        # ── 文本 token（按 class_id 索引预 tokenize 表）──────────────
        # encoder 侧：取窗口最后一帧（当前帧）的类别文本
        enc_cls   = enc_anno[-1].clamp(0, self.num_classes - 1).item()
        enc_text  = self.text_tokens[enc_cls]                # [77]

        # decoder 侧：取各预测帧对应的类别文本
        dec_cls_ids = dec_anno.clamp(0, self.num_classes - 1)   # [dec_steps]
        dec_text    = self.text_tokens[dec_cls_ids]              # [dec_steps, 77]

        # 拼合：[1+dec_steps, 77]
        text = torch.cat([enc_text.unsqueeze(0), dec_text], dim=0)

        return {
            "rgb":        enc_rgb,                                            # [enc_steps, D]
            "enc_target": enc_anno,                                           # [enc_steps]
            "dec_target": dec_anno,                                           # [dec_steps]
            "mask":       torch.ones(self.enc_steps, dtype=torch.bool),      # [enc_steps]
            "text":       text,                                               # [1+dec_steps, 77]
        }


# ────────────────────────────────────────────────────────────────────
# Collate 函数（动态 Padding）
# ────────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """动态 padding collate 函数。

    当 batch 内序列长度不一致时（如使用变长窗口），
    对 rgb 做 zero-padding，mask 标记有效位置。
    text 字段（[1+dec_steps, 77]）长度固定，直接 stack。

    Args:
        batch: list of sample dicts

    Returns:
        batched dict，所有 tensor 在 dim=0 上 stack
    """
    max_enc = max(s["rgb"].shape[0] for s in batch)
    max_dec = max(s["dec_target"].shape[0] for s in batch)

    rgb_list, enc_tgt_list, dec_tgt_list, mask_list, text_list = [], [], [], [], []

    for s in batch:
        T, D = s["rgb"].shape
        pad_len = max_enc - T

        if pad_len > 0:
            # 零填充
            rgb_padded     = F.pad(s["rgb"], (0, 0, 0, pad_len))       # [max_enc, D]
            enc_tgt_padded = F.pad(s["enc_target"], (0, pad_len), value=0)
            mask = torch.cat([s["mask"], torch.zeros(pad_len, dtype=torch.bool)])
        else:
            rgb_padded     = s["rgb"]
            enc_tgt_padded = s["enc_target"]
            mask = s["mask"]

        # dec_target padding
        dec_pad = max_dec - s["dec_target"].shape[0]
        dec_tgt_padded = (
            F.pad(s["dec_target"], (0, dec_pad), value=0) if dec_pad > 0 else s["dec_target"]
        )

        rgb_list.append(rgb_padded)
        enc_tgt_list.append(enc_tgt_padded)
        dec_tgt_list.append(dec_tgt_padded)
        mask_list.append(mask)
        # text: [1+dec_steps, 77]，长度固定，直接收集
        text_list.append(s["text"])

    return {
        "rgb":        torch.stack(rgb_list),             # [B, max_enc, D]
        "enc_target": torch.stack(enc_tgt_list),         # [B, max_enc]
        "dec_target": torch.stack(dec_tgt_list),         # [B, max_dec]
        "mask":       torch.stack(mask_list),            # [B, max_enc]
        "text":       torch.stack(text_list),            # [B, 1+dec_steps, 77]
    }


# ────────────────────────────────────────────────────────────────────
# DataLoader 构建工厂
# ────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: str | Path,
    metadata_csv: str | Path,
    enc_steps: int = 32,
    dec_steps: int = 8,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    nonzero_threshold: int = 0,
    stride: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    preload: bool = True,
    num_preload_workers: int = 8,
    bg_weight: float = 0.1,
    use_weighted_sampler: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42,
    drop_last: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """构建训练和验证 DataLoader。

    Args:
        data_dir: .pth 文件目录
        metadata_csv: metadata.csv 路径
        enc_steps: Encoder 历史帧窗口大小
        dec_steps: Decoder 预测帧数
        batch_size: 每张 GPU 的 batch size
        val_ratio: 验证集比例
        nonzero_threshold: 非背景帧最低数量
        stride: 滑动窗口步长
        num_workers: DataLoader worker 数
        pin_memory: 是否 pin memory（GPU 训练推荐开启）
        preload: 是否预加载所有特征到内存
        num_preload_workers: 预加载线程数
        bg_weight: 背景类采样/损失权重系数
        use_weighted_sampler: 是否使用 WeightedRandomSampler（过采样动作帧）
        distributed: 是否分布式训练（使用 DistributedSampler）
        world_size: 进程总数
        rank: 当前进程 rank
        seed: 随机种子
        drop_last: 是否丢弃最后不完整的 batch

    Returns:
        (train_loader, val_loader)
    """
    # ── 构建数据集 ────────────────────────────────────────────────
    train_dataset = OadFeatureDataset(
        data_dir=data_dir,
        metadata_csv=metadata_csv,
        enc_steps=enc_steps,
        dec_steps=dec_steps,
        split="train",
        val_ratio=val_ratio,
        nonzero_threshold=nonzero_threshold,
        stride=stride,
        preload=preload,
        num_preload_workers=num_preload_workers,
        bg_weight=bg_weight,
        seed=seed,
    )

    val_dataset = OadFeatureDataset(
        data_dir=data_dir,
        metadata_csv=metadata_csv,
        enc_steps=enc_steps,
        dec_steps=dec_steps,
        split="val",
        val_ratio=val_ratio,
        nonzero_threshold=0,   # 验证集不过滤无动作窗口
        stride=enc_steps,      # 验证集不重叠
        preload=preload,
        num_preload_workers=num_preload_workers,
        bg_weight=bg_weight,
        seed=seed,
    )

    # ── 构建 Sampler ──────────────────────────────────────────────
    if distributed:
        # 分布式：使用 DistributedSampler
        train_sampler: torch.utils.data.Sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=drop_last,
        )
        val_sampler: torch.utils.data.Sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    elif use_weighted_sampler:
        # 单机加权采样（过采样动作帧）
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        val_sampler = None  # 验证集顺序采样
    else:
        train_sampler = None
        val_sampler = None

    # ── 构建 DataLoader ───────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # sampler 存在时不再 shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # 验证时不存梯度，可用更大 batch
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader
