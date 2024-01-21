__all__ = ["unet_encoder3d", "unet_decoder3d"]
import math
import torch
from torch import nn, Tensor
from .transformer import RMSNorm, Attention, RotaryEmbedding, Transformer


@torch.jit.script
def fused_spatial_rmsnorm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    shape = x.shape
    x = x.view(x.shape[0], x.shape[1], -1)
    x = x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + eps) * weight.view(-1, 1) + bias.view(-1, 1)
    x = x.view(shape)
    return x


class SpatialRMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.bfloat16))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return fused_spatial_rmsnorm(hidden_states, self.scale, self.bias, self.eps)


def _naive_scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
    return attn_weight @ V


class ExportableAttention(nn.Module):
    def __init__(self, hidden_states: int, num_heads: int, head_size: int):
        super().__init__()
        self.hidden_states = hidden_states
        self.num_heads = num_heads
        self.head_size = head_size

        self.q_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=torch.bfloat16,
        )
        self.k_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=torch.bfloat16,
        )
        self.v_proj = nn.Linear(
            hidden_states,
            num_heads * head_size,
            dtype=torch.bfloat16,
        )
        self.out_proj = nn.Linear(
            num_heads * head_size,
            hidden_states,
            dtype=torch.bfloat16,
        )
        self.rotary = RotaryEmbedding(head_size)

    def forward(self, input_embeds: Tensor) -> Tensor:
        q = self.q_proj(input_embeds)
        k = self.k_proj(input_embeds)
        v = self.v_proj(input_embeds)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_size).transpose(-2, -3)
        q, k = self.rotary(q, k)
        attn_weights = _naive_scaled_dot_product_attention(q, k, v)
        attn_weights = attn_weights.transpose(-2, -3).flatten(-2)

        out = self.out_proj(attn_weights)

        return out


class ExportableAttentionLayer3d(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.norm = RMSNorm(hidden_size)
        self.attention = ExportableAttention(hidden_size, num_heads, head_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_seq = hidden_states.flatten(2).transpose(-1, -2)
        residual = input_seq

        input_seq = self.norm(input_seq)
        input_seq = self.attention(input_seq)
        input_seq = input_seq + residual
        hidden_states = input_seq.transpose(-1, -2).view_as(hidden_states)
        return hidden_states


class AttentionLayer3d(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.norm = RMSNorm(hidden_size)
        self.attention = Attention(hidden_size, num_heads, head_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_seq = hidden_states.flatten(2).transpose(-1, -2)
        residual = input_seq

        input_seq = self.norm(input_seq)
        input_seq = self.attention(input_seq)
        input_seq = input_seq + residual
        hidden_states = input_seq.transpose(-1, -2).view_as(hidden_states)
        return hidden_states


class UnetEncoderLayer3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = SpatialRMSNorm(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = SpatialRMSNorm(out_channels)
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
        self.downsample = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=2,
            stride=2,
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
        input_embeds = self.downsample(input_embeds)
        return input_embeds


class UnetEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        exportable: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        _in_channels = [int(base_channels * multiplier**i) for i in range(num_layers)]
        _out_channels = [int(base_channels * multiplier**i) for i in range(1, num_layers + 1)]
        num_heads = _out_channels[-1] // 128
        self.layers = nn.Sequential()
        self.layers.append(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=1,
            )
        )

        for i in range(num_layers):
            self.layers.append(UnetEncoderLayer3d(_in_channels[i], _out_channels[i]))
        self.layers.append(
            AttentionLayer3d(_out_channels[-1], num_heads, 128)
            if not exportable
            else ExportableAttentionLayer3d(_out_channels[-1], num_heads, 128)
        )
        self.layers.append(SpatialRMSNorm(_out_channels[-1]))
        self.layers.append(nn.SiLU(True))
        self.layers.append(
            nn.Conv3d(
                _out_channels[-1],
                latent_dim,
                kernel_size=1,
            )
        )

    def forward(self, voxel_inputs: Tensor) -> Tensor:
        return self.layers(voxel_inputs)


class UnetDecoderLayer3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = SpatialRMSNorm(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = SpatialRMSNorm(out_channels)
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
        self.upsample = nn.ConvTranspose3d(
            out_channels,
            out_channels,
            kernel_size=2,
            stride=2,
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
        input_embeds = self.upsample(input_embeds)
        return input_embeds


class UnetDecoder3d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        exportable: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        _in_channels = [int(base_channels * multiplier**i) for i in range(num_layers, 0, -1)]
        _out_channels = [int(base_channels * multiplier**i) for i in range(num_layers - 1, -1, -1)]
        num_heads = _in_channels[0] // 128

        self.layers = nn.Sequential()
        self.layers.append(
            nn.Conv3d(
                latent_dim,
                _in_channels[0],
                kernel_size=1,
            )
        )
        self.layers.append(
            AttentionLayer3d(
                _in_channels[0],
                num_heads,
                128,
            )
            if not exportable
            else ExportableAttentionLayer3d(_in_channels[0], num_heads, 128)
        )
        for i in range(num_layers):
            self.layers.append(UnetDecoderLayer3d(_in_channels[i], _out_channels[i]))
        self.layers.append(SpatialRMSNorm(base_channels))
        self.layers.append(nn.SiLU(True))
        self.layers.append(
            nn.Conv3d(
                base_channels,
                out_channels,
                kernel_size=1,
            )
        )

    def forward(self, voxel_inputs: Tensor) -> Tensor:
        return self.layers(voxel_inputs)


class Unet3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        exportable: bool = False,
    ):
        super().__init__()
        self.encoder = UnetEncoder3d(
            in_channels, latent_dim, base_channels, multiplier, num_layers, exportable=exportable
        )
        self.decoder = UnetDecoder3d(
            out_channels, latent_dim, base_channels, multiplier, num_layers, exportable=exportable
        )

    def forward(self, voxel_inputs: Tensor) -> Tensor:
        latent = self.encoder(voxel_inputs)
        return self.decoder(latent)


class UnetLatentAttention3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        base_channels: int = 64,
        multiplier: int = 2,
        num_layers: int = 3,
        num_attention_layers: int = 6,
        num_heads: int = 8,
        head_size: int = 128,
    ):
        super().__init__()
        self.encoder = UnetEncoder3d(in_channels, latent_dim, base_channels, multiplier, num_layers)
        self.decoder = UnetDecoder3d(out_channels, latent_dim, base_channels, multiplier, num_layers)
        self.latent_transformer = Transformer(latent_dim, num_attention_layers, num_heads, head_size)

    def forward(self, voxel_inputs: Tensor) -> Tensor:
        latent = self.encoder(voxel_inputs)
        latent = self.latent_transformer(latent.flatten(2).transpose(-1, -2)).transpose(-1, -2).view_as(latent)
        return self.decoder(latent)


def unet_encoder3d(
    in_channels: int,
    latent_dim: int,
    base_channels: int = 64,
    multiplier: int = 2,
    num_layers: int = 3,
) -> UnetEncoder3d:
    return UnetEncoder3d(in_channels, latent_dim, base_channels, multiplier, num_layers)


def unet_decoder3d(
    out_channels: int,
    latent_dim: int,
    base_channels: int = 64,
    multiplier: int = 2,
    num_layers: int = 3,
) -> UnetDecoder3d:
    return UnetDecoder3d(out_channels, latent_dim, base_channels, multiplier, num_layers)
