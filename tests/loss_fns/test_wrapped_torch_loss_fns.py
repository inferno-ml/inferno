import numpy.testing as npt
import torch
from torch import nn

from inferno import loss_fns

import pytest


@pytest.mark.parametrize(
    "inferno_loss_fn,torch_loss_fn,inputs,targets",
    [
        (
            loss_fns.MSELoss(),
            nn.MSELoss(),
            torch.ones((5, 10)),
            2 * torch.ones((10,)),
        ),
        (
            loss_fns.MSELoss(),
            nn.MSELoss(),
            torch.ones((5, 4, 10, 1)),
            2 * torch.ones((10, 1)),
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
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_allows_computing_loss_with_samples(
    inferno_loss_fn, torch_loss_fn, inputs, targets
):
    inferno_loss = inferno_loss_fn(inputs, targets)

    num_extra_dims = inputs.ndim - targets.ndim

    if not (
        torch.is_floating_point(targets)
        or torch.is_complex(targets)
        or inputs.ndim == targets.ndim
    ):
        num_extra_dims = num_extra_dims - 1

    torch_loss = torch_loss_fn(
        inputs.flatten(0, num_extra_dims),
        targets.expand(
            *inputs.shape[0:num_extra_dims], *(targets.ndim * (-1,))
        ).flatten(0, num_extra_dims),
    )
    npt.assert_allclose(
        inferno_loss.detach().numpy(),
        torch_loss.detach().numpy(),
    )


@pytest.mark.parametrize(
    "inferno_loss_fn,torch_loss_fn,inputs,targets",
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
def test_equivalent_to_torch_loss_fn(inferno_loss_fn, torch_loss_fn, inputs, targets):
    npt.assert_allclose(
        inferno_loss_fn(inputs, targets).detach().numpy(),
        torch_loss_fn(inputs, targets).detach().numpy(),
    )
