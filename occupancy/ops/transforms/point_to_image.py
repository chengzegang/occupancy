import math
import torch
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
import roma


def project_on(depth_image: Tensor, intrinsic: Tensor, rotation: Tensor, translation: Tensor) -> Tensor:
    """
    depth_image: (H, W)
    intrinsic: (3, 3)
    rotation: (3, 3)
    translation: (3,)
    """
    i, j = torch.meshgrid(torch.arange(depth_image.shape[0]), torch.arange(depth_image.shape[1]), indexing="ij")
    i = i.float()
    j = j.float()
    k = depth_image

    ijk = torch.stack([i, j, k], dim=0).view(3, -1)
    N = ijk.shape[-1]
    ijk = torch.matmul(torch.inverse(intrinsic), ijk)
    ijk = roma.quat_action(rotation.expand(N, -1), ijk.T) + translation
    ijk = ijk.T
    return ijk


def view_on(
    image: Tensor,
    points: Tensor,
    intrinsic: Tensor,
    rotation: Tensor,
    translation: Tensor,
    fill_value: float = 0,
    i_scale: float = 1,
    j_scale: float = 1,
) -> Tensor:
    """
    image(3, H, W)
    points: (3, N)
    intrinsic: (3, 3)
    rotation: (3, 3)
    translation: (3,)

    Projects points from world space to image space
    """
    r = torch.norm(points, dim=0, p=2)
    # transform = torch.eye(4, device=points.device, dtype=points.dtype)
    # transform[0:3, 0:3] = rotation.t()
    # transform[0:3, 3] = -translation.squeeze()
    N = points.shape[-1]
    points = roma.quat_action(roma.quat_inverse(rotation).expand(N, -1), points.T) - translation
    points = points.T
    points = torch.matmul(intrinsic, points)

    # check if points are in front of camera
    z = points[2, :]
    mask = z > 0
    points = points[:, mask]
    points = points / points[2:3, :]
    r = r[mask]

    # check if points are in image

    points[0, :] = points[0, :] * j_scale
    points[1, :] = points[1, :] * i_scale
    x = points[0, :]
    y = points[1, :]

    mask = (x >= 0) & (x < image.shape[-1]) & (y >= 0) & (y < image.shape[-2])
    points = points[:, mask]

    r = r[mask]

    # convert to pixel coordinates
    depth = torch.full_like(image[[0], :, :], fill_value, device=image.device, dtype=image.dtype)

    points = points[:3, :].floor_().long()
    ind = points[1, :] * depth.shape[-1] + points[0, :]
    shape = depth.shape

    depth = depth.view(-1)
    depth = depth.scatter_reduce_(0, ind, r, "mean", include_self=False)

    depth = depth.view(shape)
    image = torch.cat([image, depth], dim=0)

    return image
