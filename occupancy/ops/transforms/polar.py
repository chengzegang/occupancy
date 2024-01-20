import math
from typing import Tuple
from torch import Tensor
import torch
import torch.nn.functional as F


@torch.jit.script
def cartesian_to_polar(x: Tensor, y: Tensor, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(z / r)
    phi = torch.atan2(y, x)
    return r, theta, phi


@torch.jit.script
def polar_to_cartesian(r: Tensor, theta: Tensor, phi: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    # r: 0~1
    # theta: -pi~pi
    # phi: -pi/2~pi/2
    x = r * torch.cos(phi) * torch.cos(theta)
    y = r * torch.cos(phi) * torch.sin(theta)
    z = r * torch.sin(phi) * torch.sign(phi)

    return x, y, z


@torch.jit.script
def view_as_polar(
    pointcloud: Tensor, out_shape: Tuple[int, int, int] = (256, 256, 256), mode: str = "nearest", padding_mode: str = "zeros", align_corners: bool = True
):
    r, theta, phi = torch.meshgrid(
        torch.linspace(0, 1, out_shape[0]),
        torch.linspace(-torch.pi, torch.pi, out_shape[1]),
        torch.linspace(-torch.pi / 2, torch.pi / 2, out_shape[2]),
        indexing="ij",
    )
    x, y, z = polar_to_cartesian(r, theta, phi)
    grid = torch.stack([x, y, z], dim=-1)
    grid = grid.unsqueeze(0).expand(pointcloud.shape[0], -1, -1, -1, -1)

    polar_view = F.grid_sample(pointcloud, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return polar_view


@torch.jit.script
def view_as_cartesian(
    pointcloud: Tensor,
    out_shape: Tuple[int, int, int] = (256, 256, 256),
    mode: str = "nearest",
    padding_mode: str = "zeros",
    align_corners: bool = False,
):
    x, y, z = torch.meshgrid(
        torch.linspace(-1, 1, out_shape[0], device=pointcloud.device, dtype=torch.float32),
        torch.linspace(-1, 1, out_shape[1], device=pointcloud.device, dtype=torch.float32),
        torch.linspace(-1, 1, out_shape[2], device=pointcloud.device, dtype=torch.float32),
        indexing="ij",
    )
    r, theta, phi = cartesian_to_polar(x, y, z)
    r = r / math.sqrt(3) * 2 - 1
    theta = theta / torch.pi * 2 - 1
    phi = phi / torch.pi

    grid = torch.stack([r, theta, phi], dim=-1)
    grid = grid.unsqueeze(0).expand(pointcloud.shape[0], -1, -1, -1, -1)

    polar_view = F.grid_sample(pointcloud.type(torch.float32), grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners).type_as(pointcloud)
    return polar_view
