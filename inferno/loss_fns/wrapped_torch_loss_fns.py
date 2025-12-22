"""Wrappers for torch loss functions to ensure compatibility with models that sample a set of predictions."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


def predictions_and_expanded_targets(preds, targets):
    """Ensure loss can be computed with additional dimensions of (sampled) predictions.

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


class MSELoss(nn.MSELoss):

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return super().forward(*predictions_and_expanded_targets(pred, target))


class L1Loss(nn.L1Loss):

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return super().forward(*predictions_and_expanded_targets(pred, target))


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return super().forward(*predictions_and_expanded_targets(pred, target))


class NLLLoss(nn.NLLLoss):

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return super().forward(*predictions_and_expanded_targets(pred, target))


class BCELoss(nn.BCELoss):

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return super().forward(*predictions_and_expanded_targets(pred, target))


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

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

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return super().forward(*predictions_and_expanded_targets(pred, target))
