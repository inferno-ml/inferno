from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from .wrapped_torch_loss_fns import predictions_and_expanded_targets

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

__all__ = ["FocalLoss"]


class FocalLoss(nn.Module):
    r"""The focal loss rescales the cross entropy loss with a factor that induces a regularizer on the output class probabilities.

    It is useful to address class imbalance ([Lin et al. 2017](https://arxiv.org/abs/1708.02002)) and to improve
    calibration ([Mukhoti et al. 2020](http://arxiv.org/abs/2002.09437)). The loss on a single datapoint is given by

    $$
        \begin{equation*}
        \ell_n = -(1-\hat{p}_{y_n})^\gamma\log \hat{p}_{y_n}
        \end{equation*}
    $$

    For $\gamma=1$ the focal loss equals the cross entropy loss with an entropic regularizer on the predicted class probabilities.

    :param task: Specifies the type of task: 'binary' or 'multiclass'.
    :param gamma: Focusing parameter, controls the strength of the modulating factor $(1-\hat{p}_{y_n})^\gamma$.
    :param num_classes: Number of classes (only required for multi-class classification)
    :param weight: A manual rescaling weight given to each class. If given, has to be a Tensor of size C.
    :param reduction: Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied,
        ``'mean'``: the weighted mean of the output is taken,
        ``'sum'``: the output will be summed.
    """

    def __init__(
        self,
        task: Literal["binary", "multiclass"],
        gamma: float = 2.0,
        num_classes: int | None = None,
        weight: Tensor | None = None,
        reduction: Literal["none", "sum", "mean"] = "mean",
    ):
        super().__init__()
        self.task = task
        self.gamma = gamma
        self.num_classes = num_classes
        self.weight = weight
        self.reduction = reduction
        if reduction not in ["none", "sum", "mean"]:
            raise ValueError(
                f"Unsupported reduction '{self.reduction}'. Use 'none', 'sum', or 'mean'."
            )

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ):
        if self.task_type == "binary":
            return self._binary_focal_loss(pred, target)
        elif self.task_type == "multiclass":
            return self._multi_class_focal_loss(pred, target)
        else:
            raise ValueError(
                f"Unsupported task '{self.task}'. Use 'binary' or 'multiclass'."
            )

    def _binary_focal_loss(self, pred: Tensor, target: Tensor):
        """Focal loss for binary classification."""

        # Compute binary cross entropy
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            *predictions_and_expanded_targets(pred, target), reduction="none"
        )
        probs = nn.functional.sigmoid(pred)

        # Compute focal weight
        target_probs = probs * target + (1 - probs) * (1 - target)
        focal_weight = (1 - target_probs) ** self.gamma

        # Apply optional class weighting
        if self.weight is not None:
            weight = torch.as_tensor(self.weight).to(pred.device)
            weight_t = weight * target + (1 - self.weight) * (1 - target)
            bce_loss = weight_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight.flatten() * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        # TODO: ensure shape is correct when applying no reduction (sample, batch)

        return loss

    def _multi_class_focal_loss(self, pred: Tensor, target: Tensor):
        """Focal loss for multi-class classification."""

        # Compute cross-entropy for each class
        probs = nn.functional.softmax(pred, dim=-1)
        target_probs = torch.gather(
            probs, dim=-1, index=target.view(-1, 1)
        )  # TODO: ensure target view is correct even when probs contains samples
        ce_loss = nn.functional.cross_entropy(
            *predictions_and_expanded_targets(pred, target),
            reduction="none",
        )

        # Compute focal weight
        focal_weight = (1 - target_probs) ** self.gamma

        # Apply optional class weighting
        if self.weight is not None:
            weight = torch.as_tensor(self.weight).to(pred.device)
            weight_t = weight * target + (1 - self.weight) * (1 - target)
            ce_loss = weight_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.flatten() * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        # TODO: ensure shape is correct when applying no reduction (sample, batch, class)

        return loss
