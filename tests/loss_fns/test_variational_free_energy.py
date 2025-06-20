import numpy.testing as npt
import torch
from torch import nn

import inferno
from inferno import bnn, loss_fns, models
from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "model,loss_fn,inputs,targets",
    [
        (
            bnn.Sequential(
                models.MLP(
                    in_size=100,
                    hidden_sizes=[64, 64],
                    out_size=1,
                    cov=params.DiagonalCovariance(),
                ),
                nn.Flatten(-2, -1),
            ),
            nn.BCEWithLogitsLoss,
            torch.randn(32, 100),
            torch.randint(0, 10, (32,)).float(),
        ),
        (
            models.MLP(
                in_size=100,
                hidden_sizes=[64, 64],
                out_size=10,
                cov=params.DiagonalCovariance(),
            ),
            nn.CrossEntropyLoss,
            torch.randn(32, 100),
            torch.randint(0, 1, (32,)),
        ),
        (
            models.ResNet18(
                out_size=10,
                architecture="cifar",
                cov=params.DiagonalCovariance(),
            ),
            nn.CrossEntropyLoss,
            torch.randn(32, 3, 32, 32),
            torch.randint(0, 10, (32,)),
        ),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_vfe_for_different_models(model, loss_fn, inputs, targets):
    """Test whether the VFE loss function works for different models."""
    torch.manual_seed(0)

    for variational_loss_fn in [loss_fns.VariationalFreeEnergy, loss_fns.NegativeELBO]:
        variational_free_energy = variational_loss_fn(
            nll=loss_fn(),
            model=model,
        )

        variational_free_energy(model(inputs), targets)
