__all__ = ["unet_encoder2d", "unet_decoder2d"]
import torch
from torch import nn, Tensor
from .transformer import RMSNorm, Attention


class SpatialRMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, hidden_states: Tensor) -> Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.view(shape[0], shape[1], -1)
        hidden_states = hidden_states * torch.rsqrt((hidden_states**2).mean(dim=-1, keepdim=True) + self.eps)
        out = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1) * self.scale.unsqueeze(-1)
        out = out.view(shape)
        return out


class AttentionLayer2d(nn.Module):
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


class UnetEncoderLayer2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = SpatialRMSNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = SpatialRMSNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shorcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False)
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


class UnetEncoder2d(nn.Module):
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
        self.layers.append(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=1,
                bias=False
            )
        )

        for i in range(num_layers):
            self.layers.append(UnetEncoderLayer2d(_in_channels[i], _out_channels[i]))
        self.layers.append(AttentionLayer2d(_out_channels[-1], num_heads, 128))
        self.layers.append(SpatialRMSNorm(_out_channels[-1]))
        self.layers.append(nn.SiLU(True))
        self.layers.append(nn.Conv2d(_out_channels[-1], latent_dim, kernel_size=1, bias=False))

    def forward(self, voxel_inputs: Tensor) -> Tensor:
        return self.layers(voxel_inputs)


class UnetDecoderLayer2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = SpatialRMSNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = SpatialRMSNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shorcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, bias=False)
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


class SwiGLU2d(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, n_channels // 4, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_channels, n_channels // 4, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(n_channels // 4, n_channels, kernel_size=3, padding=1, bias=False)
        self.nonlinear = nn.ReLU(True)

    def forward(self, input: Tensor) -> Tensor:
        residual = input
        hidden = self.norm(input)
        w1 = self.conv1(hidden)
        w2 = self.conv2(hidden)
        hidden = self.nonlinear(w1) * w2
        hidden = self.conv3(hidden)
        hidden = hidden + residual
        return hidden


class UnetDecoder2d(nn.Module):
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
        self.layers.append(nn.Conv2d(latent_dim, _in_channels[0], kernel_size=1, bias=False))
        self.layers.append(
            AttentionLayer2d(
                _in_channels[0],
                num_heads,
                128,
            )
        )
        for i in range(num_layers):
            self.layers.append(UnetDecoderLayer2d(_in_channels[i], _out_channels[i]))
        self.layers.append(SpatialRMSNorm(base_channels))
        self.layers.append(nn.SiLU(True))
        self.layers.append(nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=False))

    def forward(self, voxel_inputs: Tensor) -> Tensor:
        return self.layers(voxel_inputs)


class Unet2d(nn.Module):
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
        self.encoder = unet_encoder2d(in_channels, latent_dim, base_channels, multiplier, num_layers)
        self.decoder = unet_decoder2d(out_channels, latent_dim, base_channels, multiplier, num_layers)

    def forward(self, voxel_inputs: Tensor) -> Tensor:
        latent = self.encoder(voxel_inputs)
        return self.decoder(latent)


def unet_encoder2d(
    in_channels: int,
    latent_dim: int,
    base_channels: int = 64,
    multiplier: int = 2,
    num_layers: int = 3,
) -> UnetEncoder2d:
    return UnetEncoder2d(in_channels, latent_dim, base_channels, multiplier, num_layers)


def unet_decoder2d(
    out_channels: int,
    latent_dim: int,
    base_channels: int = 64,
    multiplier: int = 2,
    num_layers: int = 3,
) -> UnetDecoder2d:
    return UnetDecoder2d(out_channels, latent_dim, base_channels, multiplier, num_layers)
