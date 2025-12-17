"""Wrappers for torch loss functions to ensure compatibility with models that sample a set of predictions."""

import torch
from torch import Tensor, nn


def inputs_and_expanded_targets(inputs, targets):
    """Ensure loss can be computed with additional dimensions of (sampled) predictions in inputs.

    :param inputs: Inputs (predictions).
    :param targets: Targets.
    """

    if (
        torch.is_floating_point(targets)
        or torch.is_complex(targets)
        or inputs.ndim == targets.ndim
    ):
        num_extra_dims = inputs.ndim - targets.ndim
        if num_extra_dims > 0 and not (inputs.shape[num_extra_dims:] == targets.shape):
            raise ValueError(
                "Shape mismatch between input and target. "
                "This could either be caused by incorrect target shape or an incorrect target dtype."
            )
    else:
        # If targets are classes, the inputs should have one additional dimension (for probabilities)
        num_extra_dims = inputs.ndim - targets.ndim - 1

    if num_extra_dims > 0:
        targets = targets.expand(
            *inputs.shape[0:num_extra_dims], *(targets.ndim * (-1,))
        ).reshape(-1, *targets.shape[1:])

        inputs = inputs.reshape(-1, *inputs.shape[num_extra_dims + 1 :])
    elif num_extra_dims < 0:
        raise ValueError(
            f"Shapes of input and targets do not match (input.ndim={inputs.ndim}, target.ndim={targets.ndim}).",
            f" Only predictions may have extra dimensions.",
        )

    return inputs, targets


class MSELoss(nn.MSELoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))


class L1Loss(nn.L1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))


class NLLLoss(nn.NLLLoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))


class BCELoss(nn.BCELoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))
