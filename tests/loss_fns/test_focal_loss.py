from typing import Literal

import numpy as np
import torch
from torch import testing

from inferno import loss_fns

import pytest


def preds_and_targets_for_classification_problem(
    task: Literal["binary", "multiclass"],
    num_classes: int | None = None,
    batch_shape=torch.Size(),
    sample_shape=torch.Size(),
    generator: torch.Generator | None = None,
):
    batch_shape = torch.Size(batch_shape)
    sample_shape = torch.Size(sample_shape)

    if task == "binary":
        preds = torch.randn(sample_shape + batch_shape, generator=generator)
        targets = torch.bernoulli(0.5 * torch.ones(batch_shape))
    elif task == "multiclass":
        preds = torch.randn(
            sample_shape + batch_shape + (num_classes,), generator=generator
        )
        targets = torch.randint(0, num_classes, size=batch_shape)
    else:
        raise ValueError(f"Unknown task: '{task}'.")

    return preds, targets


@pytest.mark.parametrize(
    "task,weight,num_classes,sample_shape,batch_shape",
    [
        ("binary", None, None, (3, 6), (32,)),
        ("multiclass", None, 10, (3, 6), (32,)),
        ("multiclass", torch.arange(1, 11).float() / 45, 10, (3, 6), (32,)),
        ("multiclass", 0.1 * torch.ones(10), 10, (3, 6), (32,)),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_recovers_cross_entropy_loss_with_focusing_parameter_equals_zero(
    task: str,
    weight: torch.Tensor | None,
    num_classes: int | None,
    batch_shape: torch.Size,
    sample_shape: torch.Size,
    reduction: str,
):
    # Predictions and targets
    generator = torch.Generator().manual_seed(24523)
    if task == "binary":
        ce_loss_fn = loss_fns.BCEWithLogitsLoss(
            weight=weight,
            reduction=reduction,
        )
    else:
        ce_loss_fn = loss_fns.CrossEntropyLoss(
            weight=weight,
            reduction=reduction,
        )

    preds, targets = preds_and_targets_for_classification_problem(
        task=task,
        num_classes=num_classes,
        batch_shape=batch_shape,
        sample_shape=sample_shape,
        generator=generator,
    )

    # Focal loss
    focal_loss_fn = loss_fns.FocalLoss(
        task=task,
        gamma=0.0,
        num_classes=num_classes,
        reduction=reduction,
        weight=weight,
    )

    testing.assert_close(
        focal_loss_fn(preds, targets),
        ce_loss_fn(preds, targets),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("task", ["binary", "multiclass"])
@pytest.mark.parametrize("gamma", [0.0, 1.0])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_reductions(
    task: str,
    gamma: float,
    reduction: str,
):

    # Predictions and targets
    batch_shape = (32,)
    sample_shape = (3, 6)
    generator = torch.Generator().manual_seed(24523)
    if task == "binary":
        num_classes = None
    else:
        num_classes = 10

    preds, targets = preds_and_targets_for_classification_problem(
        task=task,
        num_classes=num_classes,
        batch_shape=batch_shape,
        sample_shape=sample_shape,
        generator=generator,
    )

    # Compute loss
    loss_fn = loss_fns.FocalLoss(
        task=task, gamma=gamma, num_classes=num_classes, reduction=reduction
    )
    loss = loss_fn(preds, targets)

    # Check shape and properties of reduction
    if reduction == "mean":
        assert loss.shape == ()
        testing.assert_close(
            loss,
            loss_fns.FocalLoss(
                task=task, gamma=gamma, num_classes=num_classes, reduction="sum"
            )(preds, targets)
            / np.prod(sample_shape + batch_shape),
        )
    elif reduction == "sum":
        assert loss.shape == ()
        testing.assert_close(
            loss,
            loss_fns.FocalLoss(
                task=task, gamma=gamma, num_classes=num_classes, reduction="mean"
            )(preds, targets)
            * np.prod(sample_shape + batch_shape),
        )
    else:
        # No reduction
        assert loss.shape == sample_shape + batch_shape


@pytest.mark.parametrize(
    "task,weight,num_classes,sample_shape,batch_shape",
    [
        ("binary", None, None, (3, 6), (32,)),
        ("multiclass", torch.arange(1, 11).float() / 45, 10, (3, 6), (32,)),
        ("multiclass", 0.1 * torch.ones(10), 10, (3, 6), (32,)),
    ],
)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_updating_weight_works_correctly(
    task: str,
    weight: torch.Tensor | None,
    num_classes: int | None,
    batch_shape: torch.Size,
    sample_shape: torch.Size,
    reduction: str,
):
    # Predictions and targets
    generator = torch.Generator().manual_seed(345)
    preds, targets = preds_and_targets_for_classification_problem(
        task=task,
        num_classes=num_classes,
        batch_shape=batch_shape,
        sample_shape=sample_shape,
        generator=generator,
    )

    # Focal losses
    focal_loss_fn_correct_weights = loss_fns.FocalLoss(
        task=task, weight=weight, num_classes=num_classes, reduction=reduction
    )
    focal_loss_fn_updated_weights = loss_fns.FocalLoss(
        task=task, weight=None, num_classes=num_classes, reduction=reduction
    )
    focal_loss_fn_updated_weights.weight = weight

    # Check losses are equal
    testing.assert_close(
        focal_loss_fn_correct_weights(preds, targets),
        focal_loss_fn_updated_weights(preds, targets),
        atol=1e-6,
        rtol=1e-6,
    )
