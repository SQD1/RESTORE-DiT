# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# VDT: https://github.com/RERV/VDT/blob/main/models.py
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
"""
Codes of Denoising Transformer

A Condition_TSBlock made up of:
    1. A CrossAttentionBlock (SAR as condition).
    2. A TemporalBlock
    3. A SpatialBlock.
    4. A Feed Forward Network (MLP).
"""


import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.layers import DropPath
from einops import rearrange, reduce, repeat


def modulate(x, shift, scale, T):
    N, M = x.shape[-2], x.shape[-1]
    B = scale.shape[0]
    x = rearrange(x, '(b t) n m-> b (t n) m', b=B, t=T, n=N, m=M)
    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x = rearrange(x, 'b (t n) m-> (b t) n m', b=B, t=T, n=N, m=M)
    return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DateEmbedder(nn.Module):
    """
    Modified from https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/positional_encoding.py
    """
    def __init__(self, d: int, T: int = 10000, repeat: int | None = None, offset: int = 0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * torch.div(torch.arange(offset, offset + d).float(), 2, rounding_mode='floor') / d
        )

    def forward(self, dates):

        self.denom = self.denom.to(dates.device)

        # B x T x C, where B is equal to batch_size * H * W
        sinusoid_table = (
            dates[:, :, None] / self.denom[None, None, :]
        )
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


#################################################################################
#                                 Core Denoising Transformer                    #
#################################################################################

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, SAR=None):
        B, N, C = x.shape
        # BNC -> BNH(C/H) -> BHN(C/H)
        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(SAR).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(SAR).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BHN(C/H) @ BH(C/H)N -> BHNN
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (BHNN @ BHN(C/H)) -> BHN(C/H) -> BNH(C/H) -> BNC
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            num_frames=16
    ):
        super().__init__()
        self.num_frames = num_frames
        self.norm_x = norm_layer(hidden_size)
        self.norm_SAR = norm_layer(hidden_size)
        self.spatial_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.temporal_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, c, SAR):
        # c   [B, M]
        # x   [(B T), N, M]
        # SAR [(B T), N, M]
        shift_ca, scale_ca, gate_ca = self.adaLN_modulation(c).chunk(3, dim=1)
        T = self.num_frames
        K, N, M = x.shape
        B = K // T

        x_ln = modulate(self.norm_x(x), shift_ca, scale_ca, self.num_frames)
        SAR = self.norm_SAR(SAR)

        # cross attention on spatial dimension
        res_cross = self.drop_path(self.spatial_attn(x_ln, SAR))

        # cross attention on temporal dimension
        res_cross = rearrange(res_cross, '(b t) n m -> (b n) t m', b=B, t=T, n=N, m=M)
        SAR = rearrange(SAR, '(b t) n m -> (b n) t m', b=B, t=T, n=N, m=M)
        res_cross = self.drop_path(self.temporal_attn(res_cross, SAR))

        # gate
        res_cross = rearrange(res_cross, '(b n) t m -> b (t n) m', b=B, t=T, n=N, m=M)
        res_cross = gate_ca.unsqueeze(1) * res_cross
        res_cross = rearrange(res_cross, 'b (t n) m -> (b t) n m', b=B, t=T, n=N, m=M)

        x = x + res_cross # (b t) n m

        return x

# x = torch.rand([16,128,256])
# SAR = torch.rand([16,128,256])
# c = torch.rand([1,256])
# module = CrossAttentionBlock(256, 8)
# out = module(x, c, SAR)
# print(out.shape)

class TemporalBlock(nn.Module):
    """
    A TemporalBlock with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, num_frames=16, **block_kwargs):
        super().__init__()
        # self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.temporal_norm1 = nn.LayerNorm(hidden_size)
        self.temporal_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        shift_mta, scale_mta, gate_mta = self.adaLN_modulation(c).chunk(3, dim=1)
        T = self.num_frames
        K, N, M = x.shape
        B = K // T

        x_ln = modulate(self.temporal_norm1(x), shift_mta, scale_mta, self.num_frames)

        # temporal attention
        x_ln = rearrange(x_ln, '(b t) n m -> (b n) t m', b=B, t=T, n=N, m=M)
        res_temporal = self.temporal_attn(x_ln)

        res_temporal = rearrange(res_temporal, '(b n) t m -> b (t n) m', b=B, t=T, n=N, m=M)
        res_temporal = gate_mta.unsqueeze(1) * res_temporal
        res_temporal = rearrange(res_temporal, 'b (t n) m -> (b t) n m', b=B, t=T, n=N, m=M)
        # x = rearrange(x, '(b n) t m -> (b t) n m', b=B, t=T, n=N, m=M)
        x = x + res_temporal

        return x

# block = TemporalBlock(384, 6, 10)
# x = torch.randn(20, 256, 384)
# c = torch.randn(2, 384)
# out = block(x, c)
# print(out.shape)

class SpatialBlock(nn.Module):
    """
    A SpatialBlock with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, num_frames=16, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        T = self.num_frames
        K, N, M = x.shape
        B = K // T

        attn = self.attn(modulate(self.norm1(x), shift_msa, scale_msa, self.num_frames))
        attn = rearrange(attn, '(b t) n m-> b (t n) m', b=B, t=T, n=N, m=M)
        attn = gate_msa.unsqueeze(1) * attn
        attn = rearrange(attn, 'b (t n) m-> (b t) n m', b=B, t=T, n=N, m=M)
        x = x + attn

        return x

# block = SpatialBlock(384, 6, 4.0, 10)
# x = torch.randn(20, 256, 384)
# c = torch.randn(2, 384)
# out = block(x, c)
# print(out.shape)

class ConditionTS_Block(nn.Module):
    """
    A Condition_TSBlock made up of:
    1. A CrossAttentionBlock (SAR as condition) (optional).
    2. A TemporalBlock
    3. A SpatialBlock.
    4. A Feed Forward Network (MLP).
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, num_frames=16, if_cross_attention=True,**block_kwargs):
        super().__init__()
        self.CrossAttentionBlock = CrossAttentionBlock(hidden_size, num_heads, num_frames=num_frames)
        self.TemporalBlock = TemporalBlock(hidden_size, num_heads, num_frames=num_frames)
        self.SpatialBlock = SpatialBlock(hidden_size, num_heads, num_frames=num_frames)

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        self.num_frames = num_frames
        self.ifCrossAttention = if_cross_attention
    def forward(self, x, c, SAR):

        T = self.num_frames
        K, N, M = x.shape
        B = K // T

        if self.ifCrossAttention:
            x = self.CrossAttentionBlock(x, c, SAR)
        x = self.TemporalBlock(x, c)
        x = self.SpatialBlock(x, c)

        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1)
        mlp = self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp, self.num_frames))
        mlp = rearrange(mlp, '(b t) n m-> b (t n) m', b=B, t=T, n=N, m=M)
        mlp = gate_mlp.unsqueeze(1) * mlp
        mlp = rearrange(mlp, 'b (t n) m-> (b t) n m', b=B, t=T, n=N, m=M)
        x = x + mlp

        return x

# module = ConditionTS_Block(384, 6, 4.0, 10)
# x = torch.randn(20, 256, 384)
# c = torch.randn(2, 384)
# SAR = torch.randn(20, 256, 384)
# out = module(x, c, SAR)
# print(out.shape)



class FinalLayer(nn.Module):
    """
    The final layer of SDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, num_frames):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.num_frames = num_frames

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, self.num_frames)
        x = self.linear(x)
        return x


class SDT(nn.Module):
    """
    Sequential Denoising Transformer.
    """

    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=3,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            learn_sigma=False,
            num_frames=16,
            cond_in_channels=3,
            cross_attention=True
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.cond_in_channels = cond_in_channels

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.cond_embedder = PatchEmbed(input_size, patch_size, cond_in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.date_embedder = DateEmbedder(hidden_size, T=10000)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.num_frames = num_frames
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.time_drop = nn.Dropout(p=0)
        self.cross_attention = cross_attention


        self.blocks = nn.ModuleList(
            ConditionTS_Block(hidden_size, num_heads, mlp_ratio=mlp_ratio, num_frames=self.num_frames, if_cross_attention=self.cross_attention)
            for _ in range(depth)
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, self.num_frames)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        grid_num_frames = np.arange(self.num_frames, dtype=np.float32)
        time_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], grid_num_frames)
        self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SDT blocks:
        for TSblock in self.blocks:
            if self.cross_attention:
                nn.init.constant_(TSblock.CrossAttentionBlock.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(TSblock.CrossAttentionBlock.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(TSblock.TemporalBlock.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(TSblock.TemporalBlock.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(TSblock.SpatialBlock.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(TSblock.SpatialBlock.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(TSblock.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(TSblock.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, date=None, cond=None, location=None, semantics=None):
        """
        Forward pass of SDT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        date: (B, T) tensor of dates
        cond: same with x, conditional inputs such as SAR images
        """

        B, T, C, H, W = x.shape
        # N = int(H * W / self.patch_size ** 2)
        x = x.contiguous().view(-1, C, H, W)
        x = self.x_embedder(x) + self.pos_embed  # ((B T), N, M), where N = H * W / patch_size ** 2

        # timestep embedding
        t = t.to(x.device)  # ?
        t = self.t_embedder(t)  # (B, M)
        c = t

        # if y is not None:
        #     y = self.y_embedder(y, self.training)  # (B, M)
        #     c = t + y  # (B, M)

        # date embedding:
        if date is not None:
            date = self.date_embedder(date)
            date = rearrange(date, 'b t m -> (b t) 1 m', b=B, t=T)
            x = x + date
        else:
            # if dates are not provided
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            x = x + self.time_embed

            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> (b t) n m', b=B, t=T)


        if location is not None:
            pass

        # 在这里添加SAR embedding
        if cond is not None:
            cond = cond.contiguous().view(-1, self.cond_in_channels, H, W)
            cond = self.cond_embedder(cond) + self.pos_embed  # ((B T), N, M), where N = H * W / patch_size ** 2

            if date is not None:
                cond = cond + date
            else:
                cond = rearrange(cond, '(b t) n m -> (b n) t m', b=B, t=T)
                cond = cond + self.time_embed
                cond = self.time_drop(cond)
                cond = rearrange(cond, '(b n) t m -> (b t) n m', b=B, t=T)


        for block in self.blocks:
            x = block(x, c, cond)


        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = x.view(B, T, x.shape[-3], x.shape[-2], x.shape[-1])
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of VDT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# net = SDT(depth=3, input_size=128, hidden_size=256, patch_size=4, num_heads=4, num_frames=10, cond_in_channels=3)
# input = torch.randn(2, 10, 4, 128, 128)  # (batch, frames, channels, height, width)
# cond = torch.randn(2, 10, 3, 128, 128)
# date = torch.rand(2, 10)
# t = torch.randint(0, 100, (2,)).long()
# print(t.shape)
# output = net(input, t, date=date, cond=cond)
# print(output.shape)
# print(net)
