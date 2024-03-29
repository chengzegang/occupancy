__all__ = ["attention_unet3d"]
from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
from .transformer import CrossAttention, Attention, RMSNorm, SwiGLU
import torch.nn.functional as F
from .unet_3d import SpatialRMSNorm


class TimestepTokenEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cos_freq = nn.Linear(
            1,
            hidden_size // 2,
            dtype=torch.bfloat16,
        )
        self.sin_freq = nn.Linear(
            1,
            hidden_size // 2,
            dtype=torch.bfloat16,
        )

        self.linear_in = nn.Linear(
            hidden_size,
            hidden_size,
            dtype=torch.bfloat16,
        )
        self.norm = RMSNorm(hidden_size)
        self.linear_out = nn.Linear(
            hidden_size,
            hidden_size,
            dtype=torch.bfloat16,
        )
        self.nonlinear = nn.SiLU(True)

    def forward(self, timesteps: Tensor) -> Tensor:
        cos_freq = torch.cos_(self.cos_freq(torch.exp(timesteps.view(-1, 1) / self.hidden_size)))
        sin_freq = torch.sin_(self.sin_freq(torch.exp(timesteps.view(-1, 1) / self.hidden_size)))
        freq = torch.cat([cos_freq, sin_freq], dim=-1)

        t = self.linear_in(freq)
        t = self.norm(t)
        t = self.nonlinear(t)
        t = self.linear_out(t)
        return t


class TemporalSpatialRMSNorm(nn.Module):
    def __init__(self, num_features: int, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.timestep_bias = TimestepTokenEmbedding(num_features)
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, hidden_states: Tensor, timestep: Tensor) -> Tensor:
        t_bias = self.timestep_bias(timestep)
        shape = hidden_states.shape
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1) + t_bias.view(
            hidden_states.shape[0], -1, 1
        )
        hidden_states = (
            hidden_states
            * torch.rsqrt((hidden_states**2).mean(dim=-1, keepdim=True) + self.eps)
            * self.scale.view(-1, 1)
        )
        hidden_states = hidden_states.view(shape)
        return hidden_states


class TemporalRMSNorm(nn.Module):
    def __init__(self, num_features: int, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.timestep_bias = TimestepTokenEmbedding(num_features)
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, hidden_states: Tensor, timestep: Tensor) -> Tensor:
        t_bias = self.timestep_bias(timestep)
        hidden_states = hidden_states + t_bias.view(hidden_states.shape[0], 1, -1)
        hidden_states = (
            hidden_states * torch.rsqrt((hidden_states**2).mean(dim=-1, keepdim=True) + self.eps) * self.scale
        )

        return hidden_states


class UnetConvolution3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = TemporalSpatialRMSNorm(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = TemporalSpatialRMSNorm(out_channels)
        self.conv2 = nn.Conv3d(
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

    def forward(self, input_embeds: Tensor, timestep: Tensor) -> Tensor:
        residual = self.shorcut(input_embeds)
        input_embeds = self.norm1(input_embeds, timestep)
        input_embeds = self.nonlinear(input_embeds)
        input_embeds = self.conv1(input_embeds)
        input_embeds = self.norm2(input_embeds, timestep)
        input_embeds = self.nonlinear(input_embeds)
        input_embeds = self.conv2(input_embeds)
        input_embeds = input_embeds + residual
        return input_embeds


class ConditionalAttention3d(nn.Module):
    def __init__(self, hidden_size: int, condition_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.ln1 = TemporalRMSNorm(hidden_size)
        self.self_attn = Attention(hidden_size, num_heads, head_size)

        self.ln2 = TemporalRMSNorm(hidden_size)
        self.c_proj = nn.Linear(
            condition_size,
            hidden_size,
        )
        self.c_norm = RMSNorm(hidden_size)
        self.cross_attn = CrossAttention(hidden_size, num_heads, head_size)

        self.ln3 = TemporalRMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, hidden_size * 8 // 3, hidden_size)

    def forward(
        self,
        input_embeds: Tensor,
        condition_embeds: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        shape = input_embeds.shape

        input_embeds = input_embeds.flatten(2).transpose(-1, -2)
        condition_embeds = condition_embeds.flatten(2).transpose(-1, -2)
        residual = input_embeds
        input_embeds = self.ln1(input_embeds, timestep)
        input_embeds = self.self_attn(input_embeds)
        input_embeds = input_embeds + residual

        residual = input_embeds
        input_embeds = self.ln2(input_embeds, timestep)
        cond_embeds = self.c_proj(condition_embeds)
        cond_embeds = self.c_norm(cond_embeds)
        input_embeds = self.cross_attn(input_embeds, cond_embeds)
        input_embeds = input_embeds + residual

        residual = input_embeds
        input_embeds = self.ln3(input_embeds, timestep)
        input_embeds = self.mlp(input_embeds)
        input_embeds = input_embeds + residual

        input_embeds = input_embeds.transpose(-1, -2).view(shape)
        return input_embeds


class UnetConditionalAttentionEncoderLayer3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.convolution = UnetConvolution3d(in_channels, out_channels)
        self.downsample = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.attention = ConditionalAttention3d(out_channels, condition_size, num_heads, head_size)

    def forward(self, input_embeds: Tensor, condition_embeds: Tensor, timestep: Tensor) -> Tensor:
        input_embeds = self.convolution(input_embeds, timestep)
        input_embeds = self.downsample(input_embeds)
        input_embeds = self.attention(input_embeds, condition_embeds, timestep)
        return input_embeds


class UnetConditionalAttentionEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        condition_size: int,
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
                UnetConditionalAttentionEncoderLayer3d(
                    _in_channels[i],
                    _out_channels[i],
                    condition_size,
                    _in_channels[i] // head_size,
                    head_size,
                )
            )
        self.out_attention = ConditionalAttention3d(
            _out_channels[-1], condition_size, _out_channels[-1] // head_size, head_size
        )

        self.out_norm = SpatialRMSNorm(_out_channels[-1])
        self.out_conv = nn.Conv3d(
            _out_channels[-1],
            latent_dim,
            kernel_size=1,
        )
        self.nonlinear = nn.SiLU(True)

    def forward(self, input_embeds: Tensor, cond_embeds: Tensor, timestep: Tensor) -> Tuple[Tensor, List[Tensor]]:
        input_embeds = self.in_conv(input_embeds)
        hidden_states = [input_embeds]
        for layer in self.layers:
            if isinstance(layer, UnetConditionalAttentionEncoderLayer3d):
                hidden_states.append(layer(hidden_states[-1], cond_embeds, timestep))
            else:
                hidden_states.append(layer(hidden_states[-1], cond_embeds))
        latent = self.out_attention(hidden_states[-1], cond_embeds, timestep)
        latent = self.out_norm(latent)
        latent = self.nonlinear(latent)
        latent = self.out_conv(latent)
        return latent, hidden_states


class UnetConditionalAttentionDecoderLayer3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.shortcut = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=1,
        )
        self.attention = ConditionalAttention3d(in_channels * 2, condition_size, num_heads, head_size)
        self.upsample = nn.ConvTranspose3d(
            in_channels * 2,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.convolution = UnetConvolution3d(out_channels, out_channels)

        self.shortcut.weight.data.zero_()

    def forward(self, input_embeds: Tensor, prev_embeds: Tensor, condition_embeds: Tensor, timestep: Tensor) -> Tensor:
        input_embeds = torch.cat([input_embeds, self.shortcut(prev_embeds)], dim=1)
        input_embeds = self.attention(input_embeds, condition_embeds, timestep)
        input_embeds = self.upsample(input_embeds)
        input_embeds = self.convolution(input_embeds, timestep)
        return input_embeds


class UnetConditionalAttentionDecoder3d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        condition_size: int,
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
        self.in_attention = ConditionalAttention3d(
            _in_channels[0],
            condition_size,
            _in_channels[0] // head_size,
            head_size,
        )
        for i in range(num_layers):
            self.layers.append(
                UnetConditionalAttentionDecoderLayer3d(
                    _in_channels[i], _out_channels[i], condition_size, _in_channels[i] // head_size, head_size
                )
            )
        self.out_norm = SpatialRMSNorm(_out_channels[-1])
        self.out_conv = nn.Conv3d(
            _out_channels[-1],
            out_channels,
            kernel_size=1,
        )
        self.nonlinear = nn.SiLU(True)

    def forward(self, latents: Tensor, prev_embeds: List[Tensor], condition_embeds: Tensor, timestep: Tensor) -> Tensor:
        hidden_states = self.in_conv(latents)
        hidden_states = self.in_attention(hidden_states, condition_embeds, timestep)
        for index, layer in enumerate(self.layers):
            prev = prev_embeds[index]
            if isinstance(layer, UnetConditionalAttentionDecoderLayer3d):
                hidden_states = layer(hidden_states, prev, condition_embeds, timestep)
            else:
                hidden_states = layer(hidden_states, prev, condition_embeds)
        logits = self.out_norm(hidden_states)
        logits = self.nonlinear(logits)
        logits = self.out_conv(logits)
        return logits


class UnetConditionalAttentionMiddleLayer3d(nn.Module):
    def __init__(self, n_channels: int, condition_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.attention = ConditionalAttention3d(n_channels, condition_size, num_heads, head_size)
        self.convolution = UnetConvolution3d(n_channels, n_channels)

    def forward(self, input_embeds: Tensor, condition_embeds: Tensor, timestep: Tensor) -> Tensor:
        input_embeds = self.attention(input_embeds, condition_embeds, timestep)
        input_embeds = self.convolution(input_embeds, timestep)
        return input_embeds


class UnetConditionalAttentionBottleNeck3d(nn.Module):
    def __init__(self, n_channels: int, condition_size: int, num_layers: int, head_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                UnetConditionalAttentionMiddleLayer3d(n_channels, condition_size, n_channels // head_size, head_size)
            )

    def forward(self, input_embeds: Tensor, condition_embeds: Tensor, timestep: Tensor) -> Tensor:
        for layer in self.layers:
            input_embeds = layer(input_embeds, condition_embeds, timestep)
        return input_embeds


class UnetConditionalAttention3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        condition_size: int,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        num_middle_layers: int = 3,
        head_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = UnetConditionalAttentionEncoder3d(
            in_channels,
            latent_dim,
            condition_size,
            base_channels,
            multiplier=multiplier,
            num_layers=num_layers,
            head_size=head_size,
        )
        self.middle = UnetConditionalAttentionBottleNeck3d(
            latent_dim, condition_size, num_middle_layers, head_size=head_size
        )
        self.decoder = UnetConditionalAttentionDecoder3d(
            out_channels,
            latent_dim,
            condition_size,
            base_channels,
            multiplier=multiplier,
            num_layers=num_layers,
            head_size=head_size,
        )

    def forward(self, inputs: Tensor, condition_embeds: Tensor, timestep: Tensor) -> Tensor:
        latents, hidden_states = self.encoder(inputs, condition_embeds, timestep)
        latents = self.middle(latents, condition_embeds, timestep)
        logits = self.decoder(latents, hidden_states[::-1], condition_embeds, timestep)
        return logits


def attention_unet3d(
    in_channels: int,
    out_channels: int,
    latent_dim: int,
    condition_size: int,
    base_channels: int = 64,
    multiplier: int = 2,
    num_layers: int = 3,
    num_middle_layers: int = 3,
    head_size: int = 64,
) -> UnetConditionalAttention3d:
    return UnetConditionalAttention3d(
        in_channels,
        out_channels,
        latent_dim,
        condition_size,
        base_channels,
        multiplier=multiplier,
        num_layers=num_layers,
        num_middle_layers=num_middle_layers,
        head_size=head_size,
    )
