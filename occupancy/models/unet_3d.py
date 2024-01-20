__all__ = ["unet_encoder3d", "unet_decoder3d"]
import torch
from torch import nn, Tensor
from .transformer import RMSNorm, Attention, Transformer


@torch.jit.script
def fused_spatial_rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    shape = x.shape
    x = x.view(x.shape[0], x.shape[1], -1)
    x = x * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + eps) * weight.view(-1, 1)
    x = x.view(shape)
    return x


class SpatialRMSNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features, dtype=torch.bfloat16))

    def forward(self, hidden_states: Tensor) -> Tensor:
        return fused_spatial_rmsnorm(hidden_states, self.scale, self.eps)


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
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = SpatialRMSNorm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shorcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2, bias=False)
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        _in_channels = [int(base_channels * multiplier**i) for i in range(num_layers)]
        _out_channels = [int(base_channels * multiplier**i) for i in range(1, num_layers + 1)]
        num_heads = _out_channels[-1] // 128
        self.layers = nn.Sequential()
        self.layers.append(nn.Conv3d(in_channels, base_channels, kernel_size=1, bias=False))

        for i in range(num_layers):
            self.layers.append(UnetEncoderLayer3d(_in_channels[i], _out_channels[i]))
        self.layers.append(AttentionLayer3d(_out_channels[-1], num_heads, 128))
        self.layers.append(SpatialRMSNorm(_out_channels[-1]))
        self.layers.append(nn.SiLU(True))
        self.layers.append(nn.Conv3d(_out_channels[-1], latent_dim, kernel_size=1, bias=False))

    def forward(self, voxel_inputs: Tensor) -> Tensor:
        return self.layers(voxel_inputs)


class UnetDecoderLayer3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = SpatialRMSNorm(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = SpatialRMSNorm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shorcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.upsample = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2, bias=False)
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
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        _in_channels = [int(base_channels * multiplier**i) for i in range(num_layers, 0, -1)]
        _out_channels = [int(base_channels * multiplier**i) for i in range(num_layers - 1, -1, -1)]
        num_heads = _in_channels[0] // 128

        self.layers = nn.Sequential()
        self.layers.append(nn.Conv3d(latent_dim, _in_channels[0], kernel_size=1, bias=False))
        self.layers.append(
            AttentionLayer3d(
                _in_channels[0],
                num_heads,
                128,
            )
        )
        for i in range(num_layers):
            self.layers.append(UnetDecoderLayer3d(_in_channels[i], _out_channels[i]))
        self.layers.append(SpatialRMSNorm(base_channels))
        self.layers.append(nn.SiLU(True))
        self.layers.append(nn.Conv3d(base_channels, out_channels, kernel_size=1, bias=False))

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
    ):
        super().__init__()
        self.encoder = UnetEncoder3d(in_channels, latent_dim, base_channels, multiplier, num_layers)
        self.decoder = UnetDecoder3d(out_channels, latent_dim, base_channels, multiplier, num_layers)

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
