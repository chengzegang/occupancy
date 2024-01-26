__all__ = ["attention_unet2d"]
import math
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from .transformer import CrossAttention, Attention, RMSNorm, SwiGLU
import torch.nn.functional as F
from .unet_2d import SpatialRMSNorm


class UnetConvolution2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = SpatialRMSNorm(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = SpatialRMSNorm(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.shorcut = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.nonlinear = nn.SiLU(True)

    def forward(self, input_embeds: Tensor) -> Tensor:
        residual = self.shorcut(input_embeds)
        input_embeds = self.norm1(input_embeds)
        input_embeds = self.nonlinear(input_embeds)
        input_embeds = self.conv1(input_embeds)
        input_embeds = self.norm2(input_embeds)
        input_embeds = self.nonlinear(input_embeds)
        input_embeds = self.conv2(input_embeds)
        input_embeds = input_embeds + residual
        return input_embeds


class Attention2d(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.ln1 = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, head_size)

        self.ln2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, hidden_size * 8 // 3, hidden_size)

    def forward(
        self,
        input_embeds: Tensor,
    ) -> Tensor:
        shape = input_embeds.shape

        input_embeds = input_embeds.flatten(2).transpose(-1, -2)
        residual = input_embeds
        input_embeds = self.ln1(input_embeds)
        input_embeds = self.self_attn(input_embeds)
        input_embeds = input_embeds + residual

        residual = input_embeds
        input_embeds = self.ln2(input_embeds)
        input_embeds = self.mlp(input_embeds)
        input_embeds = input_embeds + residual

        input_embeds = input_embeds.transpose(-1, -2).view(shape)
        return input_embeds


class UnetAttentionEncoderLayer2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int, head_size: int):
        super().__init__()
        self.convolution = UnetConvolution2d(in_channels, out_channels)
        self.downsample = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.attention = Attention2d(out_channels, num_heads, head_size)

        nn.init.kaiming_normal_(self.downsample.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.downsample.bias)

    def forward(self, input_embeds: Tensor) -> Tensor:
        input_embeds = self.convolution(input_embeds)
        input_embeds = self.downsample(input_embeds)
        input_embeds = self.attention(input_embeds)
        return input_embeds


class UnetAttentionEncoder2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        head_size: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        _in_channels = [int(base_channels * multiplier**i) for i in range(num_layers)]
        _out_channels = [int(base_channels * multiplier**i) for i in range(1, num_layers + 1)]

        self.in_conv = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=1,
        )
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                UnetAttentionEncoderLayer2d(
                    _in_channels[i],
                    _out_channels[i],
                    _in_channels[i] // head_size,
                    head_size,
                )
            )
        self.out_attention = Attention2d(_out_channels[-1], _out_channels[-1] // head_size, head_size)

        self.out_norm = SpatialRMSNorm(_out_channels[-1])
        self.out_conv = nn.Conv2d(
            _out_channels[-1],
            latent_dim,
            kernel_size=1,
        )
        self.nonlinear = nn.SiLU(True)

    def forward(self, input_embeds: Tensor) -> Tuple[Tensor, List[Tensor]]:
        input_embeds = self.in_conv(input_embeds)
        hidden_states = [input_embeds]
        for layer in self.layers:
            hidden_states.append(layer(hidden_states[-1]))
        latent = self.out_attention(hidden_states[-1])
        latent = self.out_norm(latent)
        latent = self.nonlinear(latent)
        latent = self.out_conv(latent)
        return latent, hidden_states


class UnetAttentionDecoderLayer2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int, head_size: int):
        super().__init__()
        self.attention = Attention2d(in_channels * 2, num_heads, head_size)
        self.upsample = nn.ConvTranspose2d(
            in_channels * 2,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.convolution = UnetConvolution2d(out_channels, out_channels)

    def forward(self, input_embeds: Tensor, prev_embeds: Tensor) -> Tensor:
        input_embeds = torch.cat([input_embeds, prev_embeds], dim=1)
        input_embeds = self.attention(input_embeds)
        input_embeds = self.upsample(input_embeds)
        input_embeds = self.convolution(input_embeds)
        return input_embeds


class UnetAttentionDecoder2d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        head_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        _in_channels = [int(base_channels * multiplier**i) for i in range(num_layers, 0, -1)]
        _out_channels = [int(base_channels * multiplier**i) for i in range(num_layers - 1, -1, -1)]

        self.layers = nn.ModuleList()

        self.in_conv = nn.Conv2d(
            latent_dim,
            _in_channels[0],
            kernel_size=1,
        )
        self.in_attention = Attention2d(
            _in_channels[0],
            _in_channels[0] // head_size,
            head_size,
        )
        for i in range(num_layers):
            self.layers.append(
                UnetAttentionDecoderLayer2d(_in_channels[i], _out_channels[i], _in_channels[i] // head_size, head_size)
            )
        self.out_norm = SpatialRMSNorm(_out_channels[-1])
        self.out_conv = nn.Conv2d(
            _out_channels[-1],
            out_channels,
            kernel_size=1,
        )
        self.nonlinear = nn.SiLU(True)

    def forward(self, latents: Tensor, prev_embeds: List[Tensor]) -> Tensor:
        hidden_states = self.in_conv(latents)
        hidden_states = self.in_attention(hidden_states)
        for index, layer in enumerate(self.layers):
            prev = prev_embeds[index]
            hidden_states = layer(hidden_states, prev)
        logits = self.out_norm(hidden_states)
        logits = self.nonlinear(logits)
        logits = self.out_conv(logits)
        return logits


class UnetAttentionMiddleLayer2d(nn.Module):
    def __init__(self, n_channels: int, num_heads: int, head_size: int):
        super().__init__()
        self.attention = Attention2d(n_channels, num_heads, head_size)
        self.convolution = UnetConvolution2d(n_channels, n_channels)

    def forward(self, input_embeds: Tensor) -> Tensor:
        input_embeds = self.attention(input_embeds)
        input_embeds = self.convolution(input_embeds)
        return input_embeds


class UnetAttentionBottleNeck2d(nn.Module):
    def __init__(self, n_channels: int, num_layers: int, head_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(UnetAttentionMiddleLayer2d(n_channels, n_channels // head_size, head_size))

    def forward(self, input_embeds: Tensor) -> Tensor:
        for layer in self.layers:
            input_embeds = layer(input_embeds)
        return input_embeds


class UnetAttention2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        num_middle_layers: int = 3,
        head_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = UnetAttentionEncoder2d(
            in_channels, latent_dim, base_channels, multiplier=multiplier, num_layers=num_layers, head_size=head_size
        )
        self.middle = (
            UnetAttentionBottleNeck2d(latent_dim, num_middle_layers, head_size=head_size)
            if num_middle_layers > 0
            else None
        )
        self.decoder = UnetAttentionDecoder2d(
            out_channels, latent_dim, base_channels, multiplier=multiplier, num_layers=num_layers, head_size=head_size
        )

    def forward(self, inputs: Tensor) -> Tensor:
        latents, hidden_states = self.encoder(inputs)
        latents = self.middle(latents) if self.middle is not None else latents
        logits = self.decoder(latents, hidden_states[::-1])
        return logits


def attention_unet2d(
    in_channels: int,
    out_channels: int,
    latent_dim: int,
    base_channels: int = 64,
    multiplier: int = 2,
    num_layers: int = 3,
    num_middle_layers: int = 3,
    head_size: int = 64,
) -> UnetAttention2d:
    return UnetAttention2d(
        in_channels,
        out_channels,
        latent_dim,
        base_channels,
        multiplier=multiplier,
        num_layers=num_layers,
        num_middle_layers=num_middle_layers,
        head_size=head_size,
    )
