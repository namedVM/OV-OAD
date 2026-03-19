"""
ZsOadCLIP：零样本在线动作检测主模型。

将 CLIP 文本编码器 + 可选图像编码器 + OadTransformer 三者组合，
实现基于视觉-语言对比的在线动作检测。
移除了 diffdist 私有库，改用 PyTorch 原生 all_gather。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .oad_transformer import OadTransformer

# ────────────────────────────────────────────────────────────────────
# 分布式工具
# ────────────────────────────────────────────────────────────────────


def dist_collect(x: torch.Tensor) -> torch.Tensor:
    """在所有 GPU 上收集 tensor，拼接后返回。

    使用 PyTorch 原生 all_gather，替换私有 diffdist 库。
    梯度通过 GatherLayer 正确传播。

    Args:
        x: 形状 [local_batch, ...]

    Returns:
        形状 [world_size * local_batch, ...]
    """
    if not dist.is_available() or not dist.is_initialized():
        return x
    return GatherLayer.apply(x)


class GatherLayer(torch.autograd.Function):
    """支持梯度回传的分布式 all_gather。

    forward 收集所有进程的 tensor；
    backward 将梯度散播回对应进程（梯度守恒）。
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:  # type: ignore[override]
        world_size = dist.get_world_size()
        out_list = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(out_list, x.contiguous())
        # 保存本地 rank 信息用于 backward
        ctx.save_for_backward(torch.tensor(dist.get_rank()))
        ctx.world_size = world_size
        return torch.cat(out_list, dim=0)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:  # type: ignore[override]
        (rank,) = ctx.saved_tensors
        batch_size = grad_output.shape[0] // ctx.world_size
        # 取出本进程对应的梯度段
        return grad_output[rank * batch_size : (rank + 1) * batch_size]


# ────────────────────────────────────────────────────────────────────
# 文本编码器（CLIP 文本分支）
# ────────────────────────────────────────────────────────────────────


class TextEncoder(nn.Module):
    """CLIP 文本编码器包装器。

    直接复用 CLIP 预训练模型中的文本分支，
    输出 EOT token 处的投影特征。
    """

    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """前向编码文本。

        Args:
            text: token 索引 [B, context_length]

        Returns:
            文本特征 [B, text_dim]
        """
        x = self.token_embedding(text).type(self.dtype)  # [B, L, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND（CLIP Transformer 的约定）
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # 取 EOT（序列中 token 值最大的位置）对应的特征并投影
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x  # [B, text_dim]


# ────────────────────────────────────────────────────────────────────
# 主模型：ZsOadCLIP
# ────────────────────────────────────────────────────────────────────


class ZsOadCLIP(nn.Module):
    """零样本在线动作检测模型（Zero-Shot Online Action Detection with CLIP）。

    训练阶段：
        - 以视觉-语言对比损失（VTC）同时训练 encoder 全局特征 和 decoder 局部特征
        - 支持冻结 CLIP backbone，仅微调 OadTransformer

    推理阶段：
        - zero_shot_pred：输入帧特征 + 预计算文本权重，输出动作相似度 logits

    Args:
        num_class: 动作类别数（含背景）
        enc_steps: Encoder 历史帧窗口大小
        decoder_query_frames: Decoder 查询帧数
        encoder_layers: OadTransformer Encoder 层数
        decoder_layers: OadTransformer Decoder 层数
        clip_model: 预训练 CLIP 模型实例
        read_from: 数据来源，"feat"（预提取特征）或 "png"（原始图像）
        zero_shot: 是否为零样本模式
        add_fuse: 是否在 mlp_head 前融合 enc+dec 特征
        use_img_encoder: 是否在线提取图像特征（read_from=="png" 时为 True）
        loss_weight_enc: Encoder 损失权重
        loss_weight_dec: Decoder 损失权重
    """

    def __init__(
        self,
        num_class: int,
        enc_steps: int,
        decoder_query_frames: int,
        encoder_layers: int,
        decoder_layers: int,
        clip_model: nn.Module,
        read_from: str = "feat",
        zero_shot: bool = True,
        add_fuse: bool = False,
        use_img_encoder: bool = False,
        loss_weight_enc: float = 1.0,
        loss_weight_dec: float = 1.0,
    ) -> None:
        super().__init__()

        self.zero_shot = zero_shot
        self.read_from = read_from

        # ── 文本编码器（复用 CLIP 文本分支）────────────────────────
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale  # 可学习温度参数
        self.dtype = clip_model.dtype

        # ── 图像编码器（可选：在线提取特征时使用）─────────────────
        if use_img_encoder or read_from == "png":
            self.img_encoder = clip_model.visual
        else:
            # 使用预提取特征，图像编码器为恒等映射
            self.img_encoder = nn.Identity()

        # ── OAD Transformer（Encoder-Decoder 主干）────────────────
        self.oad_encoder_decoder = OadTransformer(
            num_class=num_class,
            num_tokens=enc_steps,
            decoder_query_frames=decoder_query_frames,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            zero_shot=zero_shot,
            add_fuse=add_fuse,
            loss_weight_enc=loss_weight_enc,
            loss_weight_dec=loss_weight_dec,
        )

        # 对比损失（VTC）
        self.cross_entropy = nn.CrossEntropyLoss()

        self.loss_weight_enc = loss_weight_enc
        self.loss_weight_dec = loss_weight_dec

    # ──────────────────────────────────────────────────────────────
    # 特征编码
    # ──────────────────────────────────────────────────────────────

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像/特征序列，经过 OadTransformer 输出时序特征。

        Args:
            image: 预提取特征 [B, T, D] 或原始图像帧 [B, T, C, H, W]

        Returns:
            时序特征 [B, 1+dec_q, D]
        """
        if self.read_from == "png":
            # 原始图像路径：先通过 CLIP 图像编码器提取帧特征
            b, t, c, h, w = image.size()
            image = image.reshape(-1, c, h, w)  # [B*T, C, H, W]
            image_features = self.img_encoder(image)  # [B*T, D]
            image_features = image_features.view(b, t, -1)  # [B, T, D]
            inputs = (image_features, None)
        else:
            # 预提取特征路径：直接送入 OadTransformer
            inputs = (image, None)

        # OadTransformer 输出：[B, 1+dec_q, D]
        image_features = self.oad_encoder_decoder(inputs)
        return image_features

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """编码文本 token，输出文本特征。

        Args:
            text: CLIP token [B, context_length]

        Returns:
            文本特征 [B, text_dim]
        """
        return self.text_encoder(text)

    # ──────────────────────────────────────────────────────────────
    # VTC 损失
    # ──────────────────────────────────────────────────────────────

    def vtc_loss(self, image_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:
        """视觉-文本对比损失（InfoNCE / CLIP loss）。

        支持多 GPU 分布式训练：在计算 logits 前 all_gather 所有 GPU 的特征。
        本地 batch 对应全局 batch 中对角线位置的正样本。

        Args:
            image_x: 视觉特征 [B, D]（已 L2 归一化）
            text_x: 文本特征 [B, D]（已 L2 归一化）

        Returns:
            标量损失
        """
        batch_size = image_x.shape[0]

        # 全局标签：每个样本在全局 batch 中的位置即为正样本 index
        # 注意：需显式加括号，避免 Python 运算符优先级（+ 高于 if-else）
        # 导致单机时 labels 变成 tensor+tensor 而非期望的 arange
        if dist.is_available() and dist.is_initialized():
            labels = (
                torch.arange(batch_size, dtype=torch.long, device=image_x.device)
                + batch_size * dist.get_rank()
            )
        else:
            labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device)

        # L2 归一化
        image_x = F.normalize(image_x, dim=-1)
        text_x = F.normalize(text_x, dim=-1)

        # 收集全局特征
        all_text_x = dist_collect(text_x)  # [world_size*B, D]
        all_image_x = dist_collect(image_x)  # [world_size*B, D]

        # 缩放余弦相似度
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        logits_per_img = image_x @ all_text_x.t() * logit_scale  # [B, world*B]
        logits_per_text = text_x @ all_image_x.t() * logit_scale  # [B, world*B]

        # 对称交叉熵
        loss_img = self.cross_entropy(logits_per_img, labels)
        loss_text = self.cross_entropy(logits_per_text, labels)
        return 0.5 * (loss_img + loss_text)

    # ──────────────────────────────────────────────────────────────
    # 训练 / 推理入口
    # ──────────────────────────────────────────────────────────────

    def forward_train_supervised(
        self,
        image: torch.Tensor,
        enc_target: torch.Tensor,
        dec_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """有监督训练前向计算（使用类别标注，不需要文本）。

        OadTransformer（zero_shot=False）内部直接计算交叉熵损失。

        Args:
            image:      [B, T, D]  CLIP 预提取帧特征
            enc_target: [B, T]     encoder 窗口逐帧类别 id
            dec_target: [B, dec_q] decoder 预测帧类别 id

        Returns:
            {'loss_enc_vtc': ..., 'loss_dec_vtc': ...}
        """
        # 构造 one-hot targets，形状与 OadTransformer.forward 期望一致
        # OadTransformer targets: [B, 1+dec_steps, num_class]
        #   targets[:, 0, :]  — 取最后一帧（当前帧）作为 encoder 目标
        #   targets[:, 1:, :] — decoder 预测帧目标
        B = image.shape[0]
        num_class = self.oad_encoder_decoder.num_class

        # 当前帧（enc 窗口最后一帧）作为 encoder 监督信号
        current_label = enc_target[:, -1]  # [B]
        enc_target_oh = torch.zeros(B, num_class, device=image.device)
        enc_target_oh.scatter_(
            1, current_label.unsqueeze(1).clamp(0, num_class - 1), 1.0
        )

        # decoder 预测帧的 one-hot 标注
        dec_steps = dec_target.shape[1]
        dec_target_oh = torch.zeros(B, dec_steps, num_class, device=image.device)
        for t in range(dec_steps):
            dec_target_oh[:, t, :].scatter_(
                1, dec_target[:, t].unsqueeze(1).clamp(0, num_class - 1), 1.0
            )

        # targets: [B, 1+dec_steps, num_class]
        targets = torch.cat([enc_target_oh.unsqueeze(1), dec_target_oh], dim=1)

        inputs = (image, targets)
        losses = self.oad_encoder_decoder(inputs)
        return losses

    def forward_train_contrastive(
        self, image: torch.Tensor, text: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """零样本对比预训练（VTC 损失）。

        Args:
            image: [B, T, D] 或 [B, T, C, H, W]
            text: [B, 1+dec_q, context_length]
                  text[:, 0, :] 对应 encoder 全局帧文本
                  text[:, 1:, :] 对应 decoder 查询帧文本

        Returns:
            {'loss_enc_vtc': ..., 'loss_dec_vtc': ...}
        """
        image_outs = self.encode_image(image)  # [B, 1+dec_q, D]

        # 分离 encoder 输出（全局）和 decoder 输出（逐帧）
        image_enc = image_outs[:, 0, :]  # [B, D]
        image_dec = rearrange(image_outs[:, 1:, :], "b l c -> (b l) c")  # [B*dec_q, D]

        # 文本编码：enc 侧取第 0 个 token，dec 侧展平后分别编码
        text_enc = self.encode_text(text[:, 0, :])  # [B, D]
        text_dec = self.encode_text(
            rearrange(text[:, 1:, :], "b l c -> (b l) c")  # [B*dec_q, 77]
        )  # [B*dec_q, D]

        # 计算对比损失
        loss_enc = self.vtc_loss(image_enc, text_enc) * self.loss_weight_enc
        loss_dec = self.vtc_loss(image_dec, text_dec) * self.loss_weight_dec

        return {"loss_enc_vtc": loss_enc, "loss_dec_vtc": loss_dec}

    def forward_test(
        self, image: torch.Tensor, text_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """推理前向计算。

        Args:
            image:        帧特征 [B, T, D]
            text_weights: 预计算文本嵌入 [D, num_class]（零样本推理时提供）
                          有监督模式下为 None，直接返回分类 logits

        Returns:
            logits 或特征 [B, 1+dec_q, num_class/D]
        """
        if self.zero_shot and text_weights is not None:
            return self.zero_shot_pred(image, text_weights)
        # 有监督推理：OadTransformer 直接输出分类 logits
        image_outs = self.encode_image(image)
        return image_outs

    def forward(
        self,
        image: torch.Tensor,
        text: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """统一前向入口。

        Args:
            image: 帧特征/图像
            text:
                训练有监督模式: (enc_target [B,T], dec_target [B,dec_q])
                训练对比模式:   text token [B, 1+dec_q, 77]
                推理模式:       text_weights [D, num_class] 或 None
        """
        if self.training:
            if not self.zero_shot:
                # 有监督训练
                assert isinstance(text, tuple) and len(text) == 2, (
                    "有监督训练需要传入 (enc_target, dec_target) 元组"
                )
                enc_target, dec_target = text
                return self.forward_train_supervised(image, enc_target, dec_target)
            else:
                # 零样本对比预训练
                return self.forward_train_contrastive(image, text)
        else:
            return self.forward_test(image, text)

    @torch.no_grad()
    def zero_shot_pred(
        self, image: torch.Tensor, text_weights: torch.Tensor
    ) -> torch.Tensor:
        """零样本预测：输出逐帧动作相似度 logits。

        Args:
            image: 帧特征 [B, T, D]
            text_weights: 预计算文本嵌入 [D, num_class]（已 L2 归一化）

        Returns:
            logits [B, 1+dec_q, num_class]
        """
        image_features = self.encode_image(image)  # [B, 1+dec_q, D]
        image_features = F.normalize(image_features, dim=-1)
        # 余弦相似度 × 温度缩放
        logits = (
            self.logit_scale.exp() * image_features @ text_weights
        )  # [B, 1+dec_q, num_class]
        return logits


# ────────────────────────────────────────────────────────────────────
# 模型构建工厂函数
# ────────────────────────────────────────────────────────────────────


def build_model(
    clip_model: nn.Module,
    num_class: int,
    enc_steps: int,
    decoder_query_frames: int = 8,
    encoder_layers: int = 3,
    decoder_layers: int = 5,
    read_from: str = "feat",
    zero_shot: bool = True,
    add_fuse: bool = False,
    freeze_mode: str = "none",
    loss_weight_enc: float = 1.0,
    loss_weight_dec: float = 1.0,
) -> ZsOadCLIP:
    """构建并初始化 ZsOadCLIP 模型，控制参数冻结策略。

    Args:
        clip_model: 预训练 CLIP 模型
        num_class: 动作类别数
        enc_steps: Encoder 历史帧窗口
        decoder_query_frames: Decoder 查询帧数
        encoder_layers: OadTransformer Encoder 层数
        decoder_layers: OadTransformer Decoder 层数
        read_from: "feat" 或 "png"
        zero_shot: 是否零样本模式
        add_fuse: 是否融合特征
        freeze_mode: 参数冻结策略
            "none"  — 仅训练 OadTransformer
            "both"  — 全量微调（含 OadTransformer + CLIP 两支）
            "image" — 仅微调图像编码器 + OadTransformer
            "text"  — 仅微调文本编码器 + OadTransformer
        loss_weight_enc: Encoder 损失权重
        loss_weight_dec: Decoder 损失权重

    Returns:
        配置好梯度策略的 ZsOadCLIP 实例
    """
    model = ZsOadCLIP(
        num_class=num_class,
        enc_steps=enc_steps,
        decoder_query_frames=decoder_query_frames,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        clip_model=clip_model,
        read_from=read_from,
        zero_shot=zero_shot,
        add_fuse=add_fuse,
        loss_weight_enc=loss_weight_enc,
        loss_weight_dec=loss_weight_dec,
    )

    # 1. 初始状态：全部冻结
    for param in model.parameters():
        param.requires_grad_(False)

    # 2. 根据 freeze_mode 开启特定部分的梯度
    # OadTransformer (oad_encoder_decoder) 始终开启，除非 freeze_mode 特殊指定（当前业务逻辑下必选）
    for name, param in model.named_parameters():
        if name.startswith("oad_encoder_decoder"):
            param.requires_grad_(True)

    if freeze_mode == "both":
        model.text_encoder.requires_grad_(True)
        model.img_encoder.requires_grad_(True)
        model.logit_scale.requires_grad = True
    elif freeze_mode == "image":
        model.img_encoder.requires_grad_(True)
    elif freeze_mode == "text":
        model.text_encoder.requires_grad_(True)
        model.logit_scale.requires_grad = True
    elif freeze_mode == "none":
        pass  # 仅保留 oad_encoder_decoder 的梯度
    else:
        raise ValueError(f"未知的 freeze_mode: {freeze_mode}")

    # 3. 强制性分支修正（针对 Zero-Shot 逻辑）
    # 如果是非 zero-shot 模式（有监督模式），文本分支根本不参与计算，必须彻底冻结
    if not zero_shot:
        model.text_encoder.requires_grad_(False)
        model.logit_scale.requires_grad = False

    # 如果 read_from == "feat"，img_encoder 是 nn.Identity，没有参数，但也需确保其不可训练
    if read_from == "feat":
        model.img_encoder.requires_grad_(False)

    # 转为 float32（CLIP 默认 float16，训练时建议 float32）
    model.float()
    return model
