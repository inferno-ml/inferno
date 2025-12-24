import torch
from torch import nn, testing

from inferno import loss_fns

import pytest


@pytest.mark.parametrize(
    "inferno_loss_fn,torch_loss_fn,preds,targets",
    [
        (
            loss_fns.MSELoss(),
            nn.MSELoss(),
            torch.ones((5, 10)),
            torch.randn((10,), generator=torch.Generator().manual_seed(2345)),
        ),
        (
            loss_fns.MSELoss(),
            nn.MSELoss(),
            torch.ones((5, 4, 10, 1)),
            torch.randn((10, 1), generator=torch.Generator().manual_seed(8343)),
        ),
        (
            loss_fns.L1Loss(),
            nn.L1Loss(),
            torch.randn(
                (
                    4,
                    10,
                ),
                generator=torch.Generator().manual_seed(42),
            ),
            torch.randn((10,), generator=torch.Generator().manual_seed(2345)),
        ),
        (
            loss_fns.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            torch.randn((20, 10, 5), generator=torch.Generator().manual_seed(42)),
            torch.randn((10, 5), generator=torch.Generator().manual_seed(2345)),
        ),
        (
            loss_fns.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            torch.randn((3, 10, 5), generator=torch.Generator().manual_seed(42)),
            torch.empty(10, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.CrossEntropyLoss(weight=torch.arange(1, 6) / 5),
            nn.CrossEntropyLoss(weight=torch.arange(1, 6) / 5),
            torch.randn((3, 10, 5), generator=torch.Generator().manual_seed(42)),
            torch.empty(10, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.CrossEntropyLoss(weight=torch.ones(5) / 5),
            nn.CrossEntropyLoss(weight=torch.ones(5) / 5),
            torch.randn((3, 10, 5), generator=torch.Generator().manual_seed(42)),
            torch.empty(10, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.CrossEntropyLoss(torch.as_tensor([10, 2, 1]).float()),
            nn.CrossEntropyLoss(torch.as_tensor([10, 2, 1]).float()),
            torch.randn((2, 3, 10, 3), generator=torch.Generator().manual_seed(42)),
            torch.empty(10, dtype=torch.long).random_(
                3, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            torch.randn((3, 10, 5, 3, 3), generator=torch.Generator().manual_seed(42)),
            torch.empty((10, 3, 3), dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            torch.randn((3, 10, 5, 3, 3), generator=torch.Generator().manual_seed(42)),
            torch.empty((10, 3, 3), dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.NLLLoss(),
            nn.NLLLoss(),
            torch.randn((20, 10, 5), generator=torch.Generator().manual_seed(42)),
            torch.empty(10, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.NLLLoss(weight=torch.arange(1, 6) / 5),
            nn.NLLLoss(weight=torch.arange(1, 6) / 5),
            torch.randn((20, 10, 5), generator=torch.Generator().manual_seed(42)),
            torch.empty(10, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.NLLLoss(),
            nn.NLLLoss(),
            torch.randn((5, 10, 5), generator=torch.Generator().manual_seed(42)),
            torch.empty(10, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.NLLLoss(),
            nn.NLLLoss(),
            torch.randn((6, 10, 5, 3, 3), generator=torch.Generator().manual_seed(42)),
            torch.empty((10, 3, 3), dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.BCELoss(),
            nn.BCELoss(),
            torch.rand(
                (
                    5,
                    2,
                    10,
                ),
                generator=torch.Generator().manual_seed(1345),
            ),
            torch.rand((10,), generator=torch.Generator().manual_seed(783)),
        ),
        (
            loss_fns.BCEWithLogitsLoss(),
            nn.BCEWithLogitsLoss(),
            torch.randn(
                (
                    4,
                    10,
                ),
                generator=torch.Generator().manual_seed(1345),
            ),
            torch.empty(10).random_(2, generator=torch.Generator().manual_seed(3244)),
        ),
        (
            loss_fns.BCEWithLogitsLoss(),
            nn.BCEWithLogitsLoss(),
            torch.randn(
                (
                    4,
                    10,
                ),
                generator=torch.Generator().manual_seed(1345),
            ),
            torch.empty(10).random_(2, generator=torch.Generator().manual_seed(3244)),
        ),
    ],
    ids=lambda x: x.__class__.__name__,
)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_allows_computing_loss_with_samples(
    inferno_loss_fn, torch_loss_fn, preds, targets, reduction
):
    torch_loss_fn.reduction = reduction
    inferno_loss_fn.reduction = reduction

    inferno_loss = inferno_loss_fn(preds, targets)

    num_extra_dims = preds.ndim - targets.ndim

    if not (
        torch.is_floating_point(targets)
        or torch.is_complex(targets)
        or preds.ndim == targets.ndim
    ):
        num_extra_dims = num_extra_dims - 1

    # Vmap torch loss function over sample dimensions
    torch_loss_samples = torch.vmap(
        torch_loss_fn,
        in_dims=(0, None),
    )(torch.flatten(preds, 0, num_extra_dims - 1 if num_extra_dims > 0 else 0), targets)

    # Average loss over parameter samples
    if inferno_loss_fn.reduction == "sum":
        torch_loss = torch_loss_samples.sum()
    elif inferno_loss_fn.reduction == "mean":
        torch_loss = torch_loss_samples.mean()
    else:
        # reduction="none"
        if num_extra_dims > 1:
            torch_loss_samples = torch_loss_samples.unflatten(
                0, preds.shape[0:num_extra_dims]
            )
        torch_loss = torch_loss_samples

    testing.assert_close(inferno_loss, torch_loss, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "inferno_loss_fn,torch_loss_fn,preds,targets",
    [
        (
            loss_fns.MSELoss(),
            nn.MSELoss(),
            torch.randn((10,), generator=torch.Generator().manual_seed(42)),
            torch.randn((10,), generator=torch.Generator().manual_seed(2345)),
        ),
        (
            loss_fns.L1Loss(),
            nn.L1Loss(),
            torch.randn((10,), generator=torch.Generator().manual_seed(42)),
            torch.randn((10,), generator=torch.Generator().manual_seed(2345)),
        ),
        (
            loss_fns.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            torch.randn((10, 5), generator=torch.Generator().manual_seed(42)),
            torch.randn((10, 5), generator=torch.Generator().manual_seed(2345)),
        ),
        (
            loss_fns.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            torch.randn((10, 5), generator=torch.Generator().manual_seed(42)),
            torch.empty(10, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            torch.randn((10, 5), generator=torch.Generator().manual_seed(42)),
            torch.empty(10, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.CrossEntropyLoss(),
            nn.CrossEntropyLoss(),
            torch.randn((10, 5, 3, 3, 3), generator=torch.Generator().manual_seed(42)),
            torch.empty((10, 3, 3, 3), dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.NLLLoss(),
            nn.NLLLoss(),
            torch.randn((10, 5), generator=torch.Generator().manual_seed(42)),
            torch.empty(10, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.NLLLoss(),
            nn.NLLLoss(),
            torch.randn((10, 5), generator=torch.Generator().manual_seed(42)),
            torch.empty(10, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.NLLLoss(),
            nn.NLLLoss(),
            torch.randn((10, 5, 3, 3), generator=torch.Generator().manual_seed(42)),
            torch.empty((10, 3, 3), dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.BCEWithLogitsLoss(),
            nn.BCEWithLogitsLoss(),
            torch.randn(10, generator=torch.Generator().manual_seed(1345)),
            torch.empty(10).random_(2, generator=torch.Generator().manual_seed(3244)),
        ),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_equivalent_to_torch_loss_fn(inferno_loss_fn, torch_loss_fn, preds, targets):
    testing.assert_close(
        inferno_loss_fn(preds, targets),
        torch_loss_fn(preds, targets),
    )
