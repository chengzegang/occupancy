__all__ = ["attention_unet3d"]
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from .transformer import CrossAttention, Attention, RMSNorm, SwiGLU
import torch.nn.functional as F
from .unet_3d import SpatialRMSNorm


class UnetConvolution3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = SpatialRMSNorm(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv3 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.shorcut = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.nonlinear = nn.SiLU(True)

    def forward(self, input_embeds: Tensor) -> Tensor:
        residual = self.shorcut(input_embeds)
        hidden_states = self.norm1(input_embeds)
        hidden_states = self.nonlinear(self.conv1(hidden_states)) * self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class Attention3d(nn.Module):
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


class UnetAttentionEncoderLayer3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int, head_size: int):
        super().__init__()
        self.convolution = UnetConvolution3d(in_channels, out_channels)
        self.downsample = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.attention = Attention3d(out_channels, num_heads, head_size)

    def forward(self, input_embeds: Tensor) -> Tensor:
        input_embeds = self.convolution(input_embeds)
        input_embeds = self.downsample(input_embeds)
        input_embeds = self.attention(input_embeds)
        return input_embeds


class UnetAttentionEncoder3d(nn.Module):
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

        self.in_conv = nn.Conv3d(
            in_channels,
            base_channels,
            kernel_size=1,
        )
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                UnetAttentionEncoderLayer3d(
                    _in_channels[i],
                    _out_channels[i],
                    _in_channels[i] // head_size,
                    head_size,
                )
            )
        self.out_attention = Attention3d(_out_channels[-1], _out_channels[-1] // head_size, head_size)

        self.out_norm = SpatialRMSNorm(_out_channels[-1])
        self.out_conv = nn.Conv3d(
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


class ConditionalAttention3d(nn.Module):
    def __init__(self, hidden_size: int, condition_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.ln1 = RMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, head_size)

        self.ln2 = RMSNorm(hidden_size)
        self.c_proj = nn.Linear(
            condition_size,
            hidden_size,
        )
        self.c_norm = RMSNorm(hidden_size)
        self.cross_attn = CrossAttention(hidden_size, num_heads, head_size)

        self.ln3 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, hidden_size * 8 // 3, hidden_size)

    def forward(
        self,
        input_embeds: Tensor,
        condition_embeds: Tensor,
    ) -> Tensor:
        shape = input_embeds.shape

        input_embeds = input_embeds.flatten(2).transpose(-1, -2)
        condition_embeds = condition_embeds.flatten(2).transpose(-1, -2)
        residual = input_embeds
        input_embeds = self.ln1(input_embeds)
        input_embeds = self.self_attn(input_embeds)
        input_embeds = input_embeds + residual

        residual = input_embeds
        input_embeds = self.ln2(input_embeds)
        cond_embeds = self.c_proj(condition_embeds)
        cond_embeds = self.c_norm(cond_embeds)
        input_embeds = self.cross_attn(input_embeds, cond_embeds)
        input_embeds = input_embeds + residual

        residual = input_embeds
        input_embeds = self.ln3(input_embeds)
        input_embeds = self.mlp(input_embeds)
        input_embeds = input_embeds + residual

        input_embeds = input_embeds.transpose(-1, -2).view(shape)
        return input_embeds


class UnetAttentionDecoderLayer3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int, head_size: int):
        super().__init__()
        self.attention = Attention3d(in_channels * 2, num_heads, head_size)
        self.upsample = nn.ConvTranspose3d(
            in_channels * 2,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.convolution = UnetConvolution3d(out_channels, out_channels)

    def forward(self, input_embeds: Tensor, prev_embeds: Tensor) -> Tensor:
        input_embeds = torch.cat([input_embeds, prev_embeds], dim=1)
        input_embeds = self.attention(input_embeds)
        input_embeds = self.upsample(input_embeds)
        input_embeds = self.convolution(input_embeds)
        return input_embeds


class UnetAttentionDecoder3d(nn.Module):
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

        self.in_conv = nn.Conv3d(
            latent_dim,
            _in_channels[0],
            kernel_size=1,
        )
        self.in_attention = Attention3d(
            _in_channels[0],
            _in_channels[0] // head_size,
            head_size,
        )
        for i in range(num_layers):
            self.layers.append(
                UnetAttentionDecoderLayer3d(_in_channels[i], _out_channels[i], _in_channels[i] // head_size, head_size)
            )
        self.out_norm = SpatialRMSNorm(_out_channels[-1])
        self.out_conv = nn.Conv3d(
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


class UnetAttentionMiddleLayer3d(nn.Module):
    def __init__(self, n_channels: int, num_heads: int, head_size: int):
        super().__init__()
        self.attention = Attention3d(n_channels, num_heads, head_size)
        self.convolution = UnetConvolution3d(n_channels, n_channels)

    def forward(self, input_embeds: Tensor) -> Tensor:
        input_embeds = self.attention(input_embeds)
        input_embeds = self.convolution(input_embeds)
        return input_embeds


class UnetAttentionBottleNeck3d(nn.Module):
    def __init__(self, n_channels: int, num_layers: int, head_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(UnetAttentionMiddleLayer3d(n_channels, n_channels // head_size, head_size))

    def forward(self, input_embeds: Tensor) -> Tensor:
        for layer in self.layers:
            input_embeds = layer(input_embeds)
        return input_embeds


class UnetAttention3d(nn.Module):
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
        self.encoder = UnetAttentionEncoder3d(
            in_channels, latent_dim, base_channels, multiplier=multiplier, num_layers=num_layers, head_size=head_size
        )
        self.middle = UnetAttentionBottleNeck3d(latent_dim, num_middle_layers, head_size=head_size)
        self.decoder = UnetAttentionDecoder3d(
            out_channels, latent_dim, base_channels, multiplier=multiplier, num_layers=num_layers, head_size=head_size
        )

    def forward(self, inputs: Tensor) -> Tensor:
        latents, hidden_states = self.encoder(inputs)
        latents = self.middle(latents)
        logits = self.decoder(latents, hidden_states[::-1])
        return logits


def attention_unet3d(
    in_channels: int,
    out_channels: int,
    latent_dim: int,
    base_channels: int = 64,
    multiplier: int = 2,
    num_layers: int = 3,
    num_middle_layers: int = 3,
    head_size: int = 64,
) -> UnetAttention3d:
    return UnetAttention3d(
        in_channels,
        out_channels,
        latent_dim,
        base_channels,
        multiplier=multiplier,
        num_layers=num_layers,
        num_middle_layers=num_middle_layers,
        head_size=head_size,
    )
