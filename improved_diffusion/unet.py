from abc import abstractmethod

import math
import copy
from typing import Tuple

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class FiLM(nn.Module):
    """Feature-wise linear modulation."""

    def __init__(self, H:int, W:int, channel_num:int, C:int=1):
        super().__init__()
        emb_ch = H * W * C
        self.H = H
        self.W = W
        self.C = C
        self.channel_num = channel_num
        self.flat = nn.Flatten(start_dim=2)
        self.dense = nn.Linear(emb_ch, 2 * channel_num)

    def forward(self, support: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            support: [B, S, C', H', W']
            label: [B, S, 1, H, W]
        Out:
            support: [B, S, C', H', W']
        """
        B, S, *_ = label.shape
        label = self.flat(label)
        label = label.reshape((B*S, label.shape[-1]))
        emb = self.dense(label)
        emb = emb.reshape((B, S, emb.shape[-1]))
        scale, shift = th.split(emb, self.channel_num, dim=-1)

        return support * (1. + scale) + shift


class SENetBlock(nn.Module):
    '''channel-wise modulation'''
    def __init__(self, ch:int, factor:int=16) -> None:
        super().__init__()
        self.ch = ch
        self.factor = factor
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(ch, ch // factor)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ch // factor, ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, target: torch.Tensor, hs_t: list) -> torch.Tensor:
        """
        Args:
            target: [B, C1, H, W]
            hs_t: list of [B, Cn, H, W]
        Out: 
            new_target: [B, C1, H, W]
            new_hs_t: list of [B, Cn, H, W]
        """
        list_tensor = []
        list_channels = []
        list_channels.append(target.shape[1])
        # target = self.avgpool(target)
        list_tensor.append(self.avgpool(target).squeeze())
        for h in hs_t:
            list_channels.append(h.shape[1])
            # h = self.avgpool(h)
            list_tensor.append(self.avgpool(h).squeeze())
        stacked_tensor = torch.stack(list_tensor, dim=1)
        stacked_tensor = self.fc1(stacked_tensor)
        stacked_tensor = self.relu(stacked_tensor)
        stacked_tensor = self.fc2(stacked_tensor)
        stacked_tensor = self.sigmoid(stacked_tensor)
        list_tensor = torch.split(stacked_tensor, list_channels, dim=1)
        list_tensor = [t.unsqueeze(2).unsqueeze(3) for t in list_tensor]
        new_target = target * list_tensor[0]
        new_hs_t = [h * t for h, t in zip(hs_t, list_tensor[1:])]
        return new_target, new_hs_t


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class DecoderResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        # self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        h = self.in_layers(x)
        if self.use_scale_shift_norm:
            print("no time embed here")
            raise NotImplementedError
        else:
            h = self.out_layers(h)
        return self.skip_connection(x) + h
    

class DecoderBlock(nn.Module):
    ### NOTE: CrossConvolution, Convolution are needed
    def __init__(self, in_channel, out_channel, cross_channel=None, kernel_size:int=3) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.cross_channel = cross_channel or out_channel

        if isinstance(self.in_channel, int):
            self.in_channel = (self.in_channel, self.in_channel)
        if isinstance(self.in_channel, (list, tuple)):
            assert len(self.in_channel) == 2
            if self.in_channel[0] != self.in_channel[1]:
                print("WARNING: in_channel[0] != in_channel[1], check initialization of decoder block")
        else:
            raise NotImplementedError

        self.crossconv = DecoderCrossConvBlock(
            self.in_channel[0], 
            self.cross_channel,
            kernel_size=self.kernel_size,
        )
        self.conv = DecoderConvBlock(
            self.in_channel[0], 
            self.cross_channel,
            kernel_size=self.kernel_size,
        )

    def forward(self, target: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target (torch.Tensor): [B, Ct, H', W']
            support (torch.Tensor): [B, S, Cs, H', W']
        Output:
            target (torch.Tensor): [B, Co, H, W]
            support (torch.Tensor): [B, S, Co, H, W]
        """
        assert target.shape[0] == support.shape[0]
        assert target.shape[2:] == support.shape[3:]
        target, support = self.crossconv(target, support)
        target, support = self.conv(target, support)
        return target, support


class DecoderCrossConvBlock(nn.Module):
    def __init__(self, in_channel: Tuple[int, int], out_channel, kernel_size:int=3) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        if isinstance(in_channel, int):
            self.in_channel = (in_channel, in_channel)
        if isinstance(in_channel, (list, tuple)):
            assert len(in_channel) == 2
        else:
            raise NotImplementedError
        self.concat_channel = self.in_channel[0] + self.in_channel[1]
        self.conv = nn.Conv2d(
            self.concat_channel, 
            self.out_channel,
            self.kernel_size,
            stride=1,
            padding=self.kernel_size//2,
        )
        self.nonlin = nn.LeakyReLU()

    def forward(self, target: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target: [B, Ct, H, W]
            support: [B, S, Cs, H, W]
        Return:
            target: [B, Co, H, W]
            support: [B, S, Co, H, W]
        """
        B, Ct, H, W = target.shape
        _, S, Cs, _, _ = support.shape
        assert target.shape[0] == support.shape[0]
        assert target.shape[2:] == support.shape[3:]
        assert Ct + Cs == self.concat_channel
        if Ct != Cs:
            print("WARNING: Ct != Cs, the program will proceed but wrong answer will get and you should check the inputs!")
        target = target[:, None].repeat(1, S, 1, 1, 1)
        # support = support[None].repeat(B, 1, 1, 1, 1)
        concat = torch.cat([target, support], dim=2)
        concat = concat.reshape(B*S, self.concat_channel, H, W)
        out = self.conv(concat)
        out = self.nonlin(out)
        out = out.reshape(B, S, self.out_channel, H, W)
        target = out.mean(dim=1)
        support = out
        return target, support


# need to be checked:
class DecoderConvBlock(nn.Module):
    # TODO: Convolutions part 
    def __init__(self, in_channel, out_channel, kernel_size:int=3) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        if isinstance(in_channel, int):
            self.in_channel = (in_channel, in_channel)
        if isinstance(in_channel, (list, tuple)):
            assert len(in_channel) == 2
        else:
            raise NotImplementedError
        
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            padding_mode="zeros",
            bias=True,
        )
        self.nonlin = nn.LeakyReLU()

    def forward(self, target: torch.Tensor, support: torch.Tensor):
        target = self.conv(target)
        target = self.nonlin(target)
        support = self.conv(support)
        support = self.nonlin(support)
        return target, support


class Upsample2in2out(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = 2
        if use_conv:
            self.conv_t = conv_nd(2, channels, channels, 3, padding=1)
            self.conv_s = conv_nd(2, channels, channels, 3, padding=1)

    def forward(self, target, support):
        """
        Args:
            target: [B, C, H, W]
            support: [B, S, C, H, W]
        Return:
            target: [B, C, H', W']
            support: [B, S, C, H', W']
        """
        assert target.shape[1] == self.channels
        assert support.shape[2] == self.channels
        # if self.dims == 3:
        #     x = F.interpolate(
        #         x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
        #     )
        # else:
        target = F.interpolate(target, scale_factor=2, mode="nearest")
        support = F.interpolate(support, scale_factor=2, mode="nearest")
        if self.use_conv:
            target = self.conv_t(target)
            support = self.conv_s(support)
        return target, support


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class CrossConvolutionDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        original_H=None,
        original_W=None
    ) -> None:
        super().__init__()
        
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        assert original_W is not None and original_H is not None, "please give the original spatial resolution as input"
        self.original_H = original_H
        self.original_W = original_W

        assert self.num_classes is None

        # NOTE: this decoder requires time step embedding and cannot be splitted, rewrite
        # self.decoder = copy.deepcopy(temp_unet.output_blocks)
        
        input_block_chans = [model_channels]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                ch = mult * model_channels
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                input_block_chans.append(ch)
                ds *= 2
        
        ### Original decoder with ResBlocks replaced
        self.decoder = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    DecoderResBlock(
                        ch + input_block_chans.pop(),
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.decoder.append(TimestepEmbedSequential(*layers))

        ### Replace blocks(layers) in decoder with cross convolution layer
        channel_sum = ch + sum(input_block_chans)
        self.SENetBlock = SENetBlock(channel_sum)
        self.decoder = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                self.decoder.append(
                    FiLM(
                        self.original_H, 
                        self.original_W, 
                        ch
                    )
                )
                layers = [
                    DecoderBlock(
                        ch + input_block_chans.pop(),
                        model_channels * mult,
                    )
                ]
                ch = model_channels * mult
                if level and i == num_res_blocks:
                    layers.append(Upsample2in2out(ch, conv_resample))
                    ds //= 2
                self.decoder.append(TimestepEmbedSequential(*layers))

        ### The output block is retained
        self.out = nn.Sequential(
            normalization(model_channels),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
    
    def forward(self, target: torch.Tensor, support: torch.Tensor, label: torch.Tensor, hs_t: list, hs_s: list):
        '''
        Input:
            H and W is for the original spatial resolution
            H' and W' is for the encoded spatial resolution
            S is for number of images in the support set
            The batch dimension for the support and label is removed as a batch share the whole support set
            
            target: [B, C', H', W']
            support: [B, S, C', H', W'] / [S, C', H', W']
            label: [B, S, C, H, W] / [S, C, H, W]
            hs_t: List of hidden states from the target encoder
            hs_s: List of hidden states from the encoder
        Output:
            out: [B, out_channel, H, W]
        Like what is done in Unet decoder, first concatenation then pass to model
        '''
        B, *_ = target.shape
        if len(label.shape) != len(support.shape):
            print("the label shape length and support length is not consistent, please check")
            raise ValueError
        if len(support.shape) == 4:
            assert support.shape[0] == label.shape[0]
            support = support[None].repeat(B, 1, 1, 1, 1)
        elif len(support.shape) == 5:
            assert support.shape[0] == target.shape[0]
            assert support.shape[:2] == label.shape[:2]
        else:
            print("only handle support in length 4 or 5 in shape, having:", len(support.shape), support.shape)

        target, hs_t = self.SENetBlock(target, hs_t)

        for module in self.decoder:
            if isinstance(module, FiLM):
                support = module(support, label)
            else:
                hidden_t = hs_t.pop()
                hidden_s = hs_s.pop()
                cat_target = torch.cat([target, hidden_t], dim=1)
                cat_support = torch.cat([support, hidden_s], dim=1)

                target, support = module(cat_target, cat_support)
        target = target.type(target.dtype)
        return self.out(target)
        raise NotImplementedError


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)

