import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple


@torch.jit.script
def voxelize(
    points: Tensor,
    labels: Tensor,
    x_min: int = -256,
    x_max: int = 256,
    y_min: int = -256,
    y_max: int = 256,
    z_min: int = -20,
    z_max: int = 12,
    x_size: int = 512,
    y_size: int = 512,
    z_size: int = 32,
    x_offset: int = 256,
    y_offset: int = 256,
    z_offset: int = 20,
    ignore_index: int = 0,
    num_classes: int = 1,
) -> Tensor:
    voxel = torch.zeros(
        num_classes,
        x_size,
        y_size,
        z_size,
        dtype=labels.dtype,
        device=labels.device,
    )
    valid = (
        (points[0, :] >= x_min)
        & (points[0, :] < x_max)
        & (points[1, :] >= y_min)
        & (points[1, :] < y_max)
        & (points[2, :] >= z_min)
        & (points[2, :] < z_max)
    )
    x = points[0, :]
    y = points[1, :]
    z = points[2, :]
    x = x[valid]
    y = y[valid]
    z = z[valid]
    labels = labels[:, valid]
    xf = torch.clamp(x + x_offset, 0, x_size - 1).floor().long()
    yf = torch.clamp(y + y_offset, 0, y_size - 1).floor().long()
    zf = torch.clamp(z + z_offset, 0, z_size - 1).floor().long()
    xc = torch.clamp(x + x_offset, 0, x_size - 1).ceil().long()
    yc = torch.clamp(y + y_offset, 0, y_size - 1).ceil().long()
    zc = torch.clamp(z + z_offset, 0, z_size - 1).ceil().long()
    dxf = x + x_offset - xf.to(labels.dtype)
    dyf = y + y_offset - yf.to(labels.dtype)
    dzf = z + z_offset - zf.to(labels.dtype)
    dxf = dxf.view(1, -1).to(labels.dtype)
    dyf = dyf.view(1, -1).to(labels.dtype)
    dzf = dzf.view(1, -1).to(labels.dtype)
    labels = labels.long().view(1, -1)
    voxel[:, xf, yf, zf] = voxel[:, xf, yf, zf].scatter_add_(0, labels, (1 - dxf) * (1 - dyf) * (1 - dzf))
    voxel[:, xc, yf, zf] = voxel[:, xc, yf, zf].scatter_add_(0, labels, dxf * (1 - dyf) * (1 - dzf))
    voxel[:, xf, yc, zf] = voxel[:, xf, yc, zf].scatter_add_(0, labels, (1 - dxf) * dyf * (1 - dzf))
    voxel[:, xf, yf, zc] = voxel[:, xf, yf, zc].scatter_add_(0, labels, (1 - dxf) * (1 - dyf) * dzf)
    voxel[:, xc, yc, zf] = voxel[:, xc, yc, zf].scatter_add_(0, labels, dxf * dyf * (1 - dzf))
    voxel[:, xc, yf, zc] = voxel[:, xc, yf, zc].scatter_add_(0, labels, dxf * (1 - dyf) * dzf)
    voxel[:, xf, yc, zc] = voxel[:, xf, yc, zc].scatter_add_(0, labels, (1 - dxf) * dyf * dzf)
    voxel[:, xc, yc, zc] = voxel[:, xc, yc, zc].scatter_add_(0, labels, dxf * dyf * dzf)
    voxel[ignore_index] = 0
    voxel = voxel.argmax(dim=0, keepdim=True)
    return voxel
