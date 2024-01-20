import math
import torch
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
import roma


def view_on(image: Tensor, points: Tensor, intrinsic: Tensor, rotation: Tensor, translation: Tensor, fill_value: float = 0) -> Tensor:
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
    print(points.long())

    # check if points are in image
    x = points[0, :]
    y = points[1, :]
    mask = (x >= 0) & (x < image.shape[-1]) & (y >= 0) & (y < image.shape[-2])
    points = points[:, mask]
    print(torch.numel(points))
    r = r[mask]

    # convert to pixel coordinates
    points = points[0:2, :].floor_().long()

    image = torch.cat([image, torch.full_like(image[0:1, :, :], fill_value)], dim=0)
    image[:, points[1, :], points[0, :]] = r
    return image
