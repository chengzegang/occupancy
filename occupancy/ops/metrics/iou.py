import torch
from torch import Tensor


@torch.jit.script
def iou(
    prediction: Tensor,
    ground_truth: Tensor,
    num_classes: int,
    ignore_index: int = -1,
    ignore_fill_value: float = torch.nan,
) -> Tensor:
    """
    Args:
        prediction: Tensor (B, ...)
        ground_truth: Tensor (B, ...)
        num_classes: int

    Returns:
        Tensor (B, num_classes)
    """
    if prediction.dim() > 2:
        prediction = prediction.flatten(1)
    if ground_truth.dim() > 2:
        ground_truth = ground_truth.flatten(1)
    iou = torch.full(
        (prediction.shape[0], num_classes),
        ignore_fill_value,
        dtype=torch.float32,
        device=prediction.device,
    )
    for i in range(num_classes):
        if i == ignore_index:
            continue
        intersection = (prediction == i).logical_and(ground_truth == i)
        intersection = intersection.float().sum(dim=-1)
        union = (prediction == i).logical_or(ground_truth == i)
        union = union.float().sum(dim=-1)
        iou[:, i] = intersection / union
    return iou


def mean_iou(
    prediction: Tensor,
    ground_truth: Tensor,
    num_classes: int,
    ignore_index: int = -1,
    ignore_fill_value: float = torch.nan,
) -> Tensor:
    """
    Args:
        prediction: Tensor (B, ...)
        ground_truth: Tensor (B, ...)
        num_classes: int

    Returns:
        Tensor (B, num_classes)
    """
    return iou(prediction, ground_truth, num_classes, ignore_index, ignore_fill_value).nanmean(dim=-1)
