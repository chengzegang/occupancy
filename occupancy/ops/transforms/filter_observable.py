import math
import numpy as np
from torch import Tensor
from .polar import cartesian_to_polar
import torch


def filter_observable(points: Tensor, phi_eps_degree: float = 0.5, theta_eps_degree: float = 1.0) -> Tensor:
    """Filter out points that are not observable by the sensor.

    Args:
        points (Tensor): (N, 3) tensor of points in the ego frame.
        eps (float, optional): Epsilon value to prevent division by zero. Defaults to 1e-3.

    Returns:
        Tensor: (N, 3) tensor of points in the ego frame that are observable by the sensor.
    """
    epsilon_theta = theta_eps_degree / 180 * math.pi
    epsilon_phi = phi_eps_degree / 360 * math.pi
    r, theta, phi = cartesian_to_polar(points[0], points[1], points[2])
    r_max = math.sqrt(3)
    r = r * 2 / r_max - 1
    theta = theta * 2 / torch.pi - 1
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
