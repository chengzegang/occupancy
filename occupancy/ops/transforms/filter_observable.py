import math
import numpy as np
from torch import Tensor
from .polar import cartesian_to_polar
import torch


def filter_observable(points: Tensor, epsilon_degree: float = 0.5) -> Tensor:
    """Filter out points that are not observable by the sensor.

    Args:
        points (Tensor): (N, 3) tensor of points in the ego frame.
        eps (float, optional): Epsilon value to prevent division by zero. Defaults to 1e-3.

    Returns:
        Tensor: (N, 3) tensor of points in the ego frame that are observable by the sensor.
    """
    epsilon_theta = epsilon_degree / 180 * math.pi
    epsilon_phi = epsilon_degree / 360 * math.pi
    r, theta, phi = cartesian_to_polar(points[0], points[1], points[2])
    r = r / math.sqrt(3) * 2 - 1
    theta = theta / torch.pi * 2 - 1
    phi = phi / torch.pi

    view_angle = torch.stack([theta, phi], dim=0)
    view_angle[0] /= epsilon_theta
    view_angle[1] /= epsilon_phi
    view_angle = view_angle.long()
    sort_ind = torch.sort(r).indices

    view_angle = view_angle[:, sort_ind]
    uniq_ind = np.unique(view_angle.numpy(), axis=1, return_index=True)[1]
    final_ind = sort_ind[uniq_ind]

    return final_ind
