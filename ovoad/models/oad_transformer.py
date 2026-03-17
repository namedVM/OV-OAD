"""
OAD Transformer 模型实现
基于 Encoder-Decoder 架构，用于在线动作检测（Online Action Detection）。
替换废弃库（mmcv、parrots 等），使用 PyTorch >= 2.4 原生实现。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_


# ────────────────────────────────────────────────────────────────────
# 基础工具模块
# ────────────────────────────────────────────────────────────────────

class LayerScale(nn.Module):
    """LayerScale：对残差分支进行可学习的缩放（来自 CaiT 论文）。

    Args:
        dim: 特征维度
        init_values: 初始缩放值，默认 1e-5
        inplace: 是否使用 in-place 操作
    """

    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    """标准两层 MLP，用于 Transformer 的 FFN 子层。

    Args:
        in_features: 输入维度
        hidden_features: 隐层维度（默认与输入相同）
        out_features: 输出维度（默认与输入相同）
        act_layer: 激活函数类
        drop: dropout 概率
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ────────────────────────────────────────────────────────────────────
# 注意力机制
# ────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """标准多头自注意力，用于 Encoder 块。

    Args:
        dim: 特征维度
        num_heads: 注意力头数
        qkv_bias: 是否为 QKV 线性层添加偏置
        attn_drop: 注意力权重 dropout 概率
        proj_drop: 输出投影 dropout 概率
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} 必须被 num_heads={num_heads} 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # [B, N, 3*C] -> [3, B, heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # 各自 [B, heads, N, head_dim]

        # 缩放点积注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 聚合值向量并投影
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TriangularCausalMask:
    """因果掩码（上三角），用于 Decoder 自注意力以保证在线推理时不窥探未来帧。"""

    def __init__(self, B: int, L: int, device: torch.device | str = "cpu") -> None:
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # True 位置将被屏蔽（填充为 -inf）
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class CrossAttention(nn.Module):
    """通用交叉/自注意力模块，支持因果掩码。

    Args:
        dim: 特征维度
        num_heads: 注意力头数
        qkv_bias: QK 线性层是否含偏置
        cross_attn_flag: True 时执行交叉注意力（Q 来自 x，K/V 来自 y）
        mask_flag: True 时对自注意力应用因果掩码
        attn_drop: 注意力 dropout
        proj_drop: 投影 dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        cross_attn_flag: bool = False,
        mask_flag: bool = False,
        attn_drop: float = 0.1,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} 必须被 num_heads={num_heads} 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mask_flag = mask_flag
        self.cross_attn_flag = cross_attn_flag

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _project(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
        """计算 Q/K/V 投影。

        Returns:
            (q, k, v, B, N, C, L) 其中 N 为 decoder 序列长，L 为 encoder 序列长
        """
        B, N, C = x.shape
        _, L, _ = y.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim)

        if not self.cross_attn_flag:
            # 自注意力：K/V 也来自 x
            k = self.k(x).reshape(B, N, self.num_heads, self.head_dim)
            v = self.v(x).reshape(B, N, self.num_heads, self.head_dim)
        else:
            # 交叉注意力：K/V 来自 encoder 输出 y
            k = self.k(y).reshape(B, L, self.num_heads, self.head_dim)
            v = self.v(y).reshape(B, L, self.num_heads, self.head_dim)

        return q, k, v, B, N, C, L

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attn_mask: Optional[TriangularCausalMask] = None,
    ) -> torch.Tensor:
        q, k, v, B, N, C, L = self._project(x, y)

        # 计算注意力分数：einsum 等价于 bmm 但维度更清晰
        # q: [B, N, H, D], k: [B, L, H, D] -> attn: [B, H, N, L]
        attn = torch.einsum("bnhd,blhd->bhnl", q, k)

        # 应用因果掩码（训练阶段 Decoder 自注意力）
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, N, device=q.device)
            attn = attn.masked_fill(attn_mask.mask, float("-inf"))

        attn = torch.softmax(self.scale * attn, dim=-1)
        attn = self.attn_drop(attn)

        # v: [B, L, H, D] -> out: [B, N, C]
        out = torch.einsum("bhnl,blhd->bnhd", attn, v).reshape(B, N, C).contiguous()
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ────────────────────────────────────────────────────────────────────
# Transformer 基础块
# ────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """标准 Transformer Encoder 块（Pre-Norm）。

    Args:
        dim: 特征维度
        num_heads: 注意力头数
        mlp_ratio: FFN 隐层扩展比率
        qkv_bias: QKV 是否含偏置
        drop: FFN/Proj dropout
        attn_drop: 注意力 dropout
        init_values: LayerScale 初始值，None 则不使用 LayerScale
        drop_path: Stochastic Depth 概率
        act_layer: 激活函数类
        norm_layer: 归一化层类
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm + 残差
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CrossAttnBlock(Block):
    """Decoder 块：自注意力（带因果掩码）+ 交叉注意力 + FFN。

    继承自 Block，在自注意力之后插入交叉注意力子层。

    Args:
        directional_mask: 是否对自注意力应用因果掩码（在线推理需要）
        其余参数同 Block
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        directional_mask: bool = True,
    ) -> None:
        super().__init__(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop, attn_drop=attn_drop, init_values=init_values,
            drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
        )

        # 层 1：带因果掩码的自注意力（替换父类中不带掩码的 Attention）
        self.attn = CrossAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            cross_attn_flag=False, mask_flag=directional_mask,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm1_y = norm_layer(dim)  # encoder 侧 LN（自注意力阶段不使用 y，仅为接口一致）

        # 层 2：交叉注意力
        self.norm3 = norm_layer(dim)
        self.norm3_y = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            cross_attn_flag=True, mask_flag=False,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 层 3：FFN 继承自父类（norm2、mlp、ls2、drop_path2）

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = inputs
        assert y is not None, "CrossAttnBlock 需要同时提供 x（decoder）和 y（encoder）"

        # 层 1：带因果掩码的自注意力
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1_y(y), attn_mask=None)))
        # 层 2：交叉注意力（Q 来自 decoder，K/V 来自 encoder）
        x = x + self.drop_path3(self.ls3(self.cross_attn(self.norm3(x), self.norm3_y(y), attn_mask=None)))
        # 层 3：FFN
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x, y


# ────────────────────────────────────────────────────────────────────
# OAD Transformer 主干
# ────────────────────────────────────────────────────────────────────

class OadTransformer(nn.Module):
    """OAD Transformer：在线动作检测的编码器-解码器 Transformer。

    架构概述：
        1. Encoder：对历史帧特征序列（+ CLS token）做自注意力编码
        2. Decoder：以可学习的查询 token 为 Q，编码器输出为 KV，预测当前动作

    零样本模式（zero_shot=True）：
        - 输出特征向量，与文本嵌入做余弦相似度
    有监督模式（zero_shot=False）：
        - 直接输出分类 logits，使用交叉熵损失

    Args:
        num_class: 动作类别数（含背景）
        num_tokens: Encoder 输入的历史帧数（enc_steps）
        decoder_query_frames: Decoder 查询 token 数（dec_query）
        encoder_embedding_dim: Encoder 特征维度
        decoder_embedding_dim: Decoder 特征维度
        encoder_num_heads: Encoder 注意力头数
        decoder_num_heads: Decoder 注意力头数
        encoder_layers: Encoder 层数
        decoder_layers: Decoder 层数
        encoder_mlp_ratio: Encoder FFN 扩展比
        decoder_mlp_ratio: Decoder FFN 扩展比
        dropout_rate: 全局 dropout 概率
        encoder_drop_path_rate: Encoder stochastic depth 最大概率
        decoder_drop_path_rate: Decoder stochastic depth 最大概率
        decoder_attn_dp: Decoder 注意力 dropout
        encoder_attn_dp: Encoder 注意力 dropout
        qkv_bias: QKV 线性层偏置
        class_token: 是否在 Encoder 输入中加入 CLS token
        directional_mask: Decoder 是否使用因果掩码
        zero_shot: 零样本模式（输出特征）或有监督模式（输出 logits）
        add_fuse: 是否融合 enc+dec 特征（通过 MLP 投影）
    """

    def __init__(
        self,
        num_class: int,
        num_tokens: int,
        decoder_query_frames: int = 8,
        encoder_embedding_dim: int = 512,
        decoder_embedding_dim: int = 512,
        encoder_num_heads: int = 8,
        decoder_num_heads: int = 4,
        encoder_layers: int = 3,
        decoder_layers: int = 5,
        encoder_mlp_ratio: float = 4.0,
        decoder_mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
        encoder_drop_path_rate: float = 0.1,
        decoder_drop_path_rate: float = 0.1,
        decoder_attn_dp: float = 0.1,
        encoder_attn_dp: float = 0.1,
        qkv_bias: bool = True,
        class_token: bool = True,
        directional_mask: bool = True,
        zero_shot: bool = True,
        add_fuse: bool = False,
        loss_weight_enc: float = 1.0,
        loss_weight_dec: float = 1.0,
    ) -> None:
        super().__init__()

        assert encoder_embedding_dim % encoder_num_heads == 0
        assert decoder_embedding_dim % decoder_num_heads == 0

        self.zero_shot = zero_shot
        self.add_fuse = add_fuse
        self.num_class = num_class
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim

        # ── Encoder ──────────────────────────────────────────────────
        self.global_tokens = 1 if class_token else 0
        self.seq_length = num_tokens + self.global_tokens

        # CLS token（用于全局语义汇聚）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embedding_dim))

        # 位置编码（可学习，初始化为小随机值）
        self.encoder_position_encoding = nn.Parameter(
            torch.randn(1, self.seq_length, encoder_embedding_dim) * 0.02
        )
        self.encoder_pos_drop = nn.Dropout(p=dropout_rate)
        self.pre_head_ln = nn.LayerNorm(encoder_embedding_dim)

        # Stochastic Depth 线性衰减
        encoder_dpr = [x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_layers)]
        self.encoder = nn.Sequential(*[
            Block(
                dim=encoder_embedding_dim,
                num_heads=encoder_num_heads,
                mlp_ratio=encoder_mlp_ratio,
                qkv_bias=qkv_bias,
                drop=dropout_rate,
                attn_drop=encoder_attn_dp,
                drop_path=encoder_dpr[i],
            )
            for i in range(encoder_layers)
        ])

        # ── Decoder ──────────────────────────────────────────────────
        self.decoder_position_encoding = nn.Parameter(
            torch.randn(1, self.seq_length, decoder_embedding_dim) * 0.02
        )
        self.decoder_pos_drop = nn.Dropout(p=dropout_rate)

        # 可学习解码器查询 token（对应 decoder_query_frames 帧）
        self.decoder_cls_token = nn.Parameter(
            torch.zeros(1, decoder_query_frames, decoder_embedding_dim)
        )

        # Stochastic Depth 线性衰减
        decoder_dpr = [x.item() for x in torch.linspace(0, decoder_drop_path_rate, decoder_layers)]
        self.decoder = nn.Sequential(*[
            CrossAttnBlock(
                dim=decoder_embedding_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=decoder_mlp_ratio,
                qkv_bias=qkv_bias,
                drop=dropout_rate,
                attn_drop=decoder_attn_dp,
                drop_path=decoder_dpr[i],
                directional_mask=directional_mask,
            )
            for i in range(decoder_layers)
        ])

        self.after_dropout = nn.Dropout(p=dropout_rate)

        # ── 输出头 ───────────────────────────────────────────────────
        if not self.zero_shot:
            # 有监督：Decoder 输出直接分类
            self.classifier = nn.Linear(decoder_embedding_dim, self.num_class)
            self.loss_weight_enc = loss_weight_enc
            self.loss_weight_dec = loss_weight_dec
            self.cross_entropy = nn.CrossEntropyLoss()

        if self.zero_shot:
            # 零样本：拼接 enc+dec 特征后可选投影
            if self.add_fuse:
                self.mlp_head = nn.Linear(
                    encoder_embedding_dim + decoder_embedding_dim, encoder_embedding_dim
                )
            else:
                self.mlp_head = nn.Identity()
        else:
            # 有监督：拼接后分类
            self.mlp_head = nn.Linear(
                encoder_embedding_dim + decoder_embedding_dim, self.num_class
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """权重初始化：CLS token 和 pos_enc 用截断正态分布。"""
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.decoder_cls_token, std=0.02)

    def forward(
        self,
        inputs: tuple[torch.Tensor, Optional[torch.Tensor]],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """前向计算。

        Args:
            inputs: (x, targets)
                x: 历史帧特征 [B, T, D]，T=num_tokens，D=encoder_embedding_dim
                targets: 分类标签（有监督训练时使用）
                    训练时 [B, 1+dec_steps, num_class]
                    推理时 None

        Returns:
            训练时（有监督）: 损失字典 {'loss_enc_vtc': ..., 'loss_dec_vtc': ...}
            推理时（有监督）: 拼接预测 [B, 1+dec_steps, num_class]
            训练/推理时（零样本）: 特征 [B, 1+dec_steps, D]
        """
        x, targets = inputs  # x: [B, T, D]

        # ── Encoder 阶段 ──────────────────────────────────────────────
        # 追加 CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, D]
        x = torch.cat((x, cls_tokens), dim=1)  # [B, T+1, D]
        x = self.encoder_pos_drop(x + self.encoder_position_encoding)  # 加位置编码

        x = self.encoder(x)          # [B, T+1, enc_dim]
        x = self.pre_head_ln(x)      # LayerNorm

        # ── Decoder 阶段 ──────────────────────────────────────────────
        # 解码器使用可学习查询 token
        dec_query = self.decoder_cls_token.expand(x.shape[0], -1, -1)  # [B, dec_q, dec_dim]
        decoder_inputs = (dec_query, x)   # (query, encoder_output)
        dec, _ = self.decoder(decoder_inputs)  # [B, dec_q, dec_dim]
        dec = self.after_dropout(dec)

        # 全局平均池化（decoder 输出）
        dec_for_token = dec.mean(dim=1)  # [B, dec_dim]

        # ── 输出投影 ──────────────────────────────────────────────────
        if not self.zero_shot:
            # 有监督：decoder 每帧单独分类
            dec_cls_out = self.classifier(dec)                     # [B, dec_q, num_class]
            dec_cls_out_ = dec_cls_out.view(-1, self.num_class)    # [B*dec_q, num_class]

        # 融合 encoder CLS token 与 decoder 平均特征
        if self.zero_shot and not self.add_fuse:
            # 直接相加（不做 concat）
            x_fused = x[:, -1] + dec_for_token                   # [B, dim]
        else:
            # 拼接并通过 MLP 投影
            x_fused = torch.cat((x[:, -1], dec_for_token), dim=1)  # [B, enc_dim + dec_dim]

        x_fused = self.mlp_head(x_fused)  # [B, dim] 或 [B, num_class]

        # ── 返回值 ────────────────────────────────────────────────────
        if not self.zero_shot:
            if not self.training:
                # 推理：拼接 enc 和 dec 预测
                return torch.cat((x_fused.unsqueeze(1), dec_cls_out), dim=1)  # [B, 1+dec_q, num_class]
            # 训练：计算损失
            # targets[:, 0, :] 是 encoder 目标（当前帧），targets[:, 1:, :] 是 decoder 目标
            loss_enc = self.cross_entropy(x_fused, targets[:, 0, :]) * self.loss_weight_enc
            loss_dec = self.cross_entropy(
                dec_cls_out_, targets[:, 1:, :].reshape(-1, self.num_class)
            ) * self.loss_weight_dec
            return {"loss_enc_vtc": loss_enc, "loss_dec_vtc": loss_dec}
        else:
            # 零样本：返回 [B, 1+dec_q, dim] 供文本对比
            return torch.cat((x_fused.unsqueeze(1), dec), dim=1)
