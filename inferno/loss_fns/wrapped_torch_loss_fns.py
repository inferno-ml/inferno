"""Wrappers for torch loss functions to ensure compatibility with models that sample a set of predictions."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


def _num_extra_dims(preds: Tensor, targets: Tensor) -> int:
    """Compute number of extra (sampling) dimensions in `preds` over `targets` depending on the data type.

    :param preds: Predictions.
    :param targets: Targets.
    """
    if (
        torch.is_floating_point(targets)
        or torch.is_complex(targets)
        or preds.ndim == targets.ndim
    ):
        num_extra_dims = preds.ndim - targets.ndim
        if num_extra_dims > 0 and not (preds.shape[num_extra_dims:] == targets.shape):
            raise ValueError(
                "Shape mismatch between pred and target. "
                + "This could either be caused by incorrect target shape or an incorrect target dtype."
            )
    else:
        # If targets are classes, the predictions should have one additional dimension (for probabilities)
        num_extra_dims = preds.ndim - targets.ndim - 1

    return num_extra_dims


def _predictions_and_expanded_targets(
    preds: Tensor, targets: Tensor, num_extra_dims: int | None = None
):
    """Reshape and expand predictions and targets appropriately for sampled predictions.

    Ensures loss can be computed with additional dimensions of (sampled) predictions beyond a single batch dimension.

    :param preds: Predictions.
    :param targets: Targets.
    :param num_extra_dims: Number of extra (sampling) batch dimensions.
    """
    if num_extra_dims is None:
        num_extra_dims = _num_extra_dims(preds, targets)

    if num_extra_dims > 0:
        targets = targets.expand(
            *preds.shape[0:num_extra_dims], *(targets.ndim * (-1,))
        ).reshape(-1, *targets.shape[1:])

        preds = preds.reshape(-1, *preds.shape[num_extra_dims + 1 :])
    elif num_extra_dims < 0:
        raise ValueError(
            f"Shapes of pred and targets do not match (pred.ndim={preds.ndim}, target.ndim={targets.ndim}). "
            + f"Only predictions may have extra dimensions.",
        )

    return preds, targets


class MultipleBatchDimensionsLossMixin:
    """Mixin class which allows computing loss across additional batch dimensions."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Compute loss by flattening all batch dimensions into a single one
        num_extra_dims = _num_extra_dims(pred, target)
        loss = super().forward(
            *_predictions_and_expanded_targets(
                pred, target, num_extra_dims=num_extra_dims
            )
        )

        if self.reduction == "none":
            # Reshape to restore batch dimensions
            loss = loss.unflatten(0, pred.shape[0 : num_extra_dims + 1])

        return loss


class MSELoss(MultipleBatchDimensionsLossMixin, nn.MSELoss):
    pass


class L1Loss(MultipleBatchDimensionsLossMixin, nn.L1Loss):
    pass


class CrossEntropyLoss(MultipleBatchDimensionsLossMixin, nn.CrossEntropyLoss):
    pass


class NLLLoss(MultipleBatchDimensionsLossMixin, nn.NLLLoss):
    pass


class BCELoss(MultipleBatchDimensionsLossMixin, nn.BCELoss):
    pass


class BCEWithLogitsLoss(MultipleBatchDimensionsLossMixin, nn.BCEWithLogitsLoss):

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        pos_weight: Tensor | None = None,
    ):
        if weight is not None:
            raise NotImplementedError(
                "Batch 'weight' rescaling is currently not implemented in Inferno."
            )
        if pos_weight is not None:
            raise NotImplementedError(
                "'pos_weight' argument not implemented in Inferno."
            )

        super().__init__(weight=weight, reduction=reduction, pos_weight=pos_weight)
