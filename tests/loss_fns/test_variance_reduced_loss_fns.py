import numpy.testing as npt
import torch
from torch import nn

from inferno import bnn, loss_fns, models

import pytest


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize(
    "loss_fn,loss_fn_variance_reduced,model,input,target",
    [
        (
            loss_fns.MSELoss,
            loss_fns.MSELossVR,
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8, 8],
                out_size=1,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    None,
                    bnn.params.LowRankCovariance(4),
                ],
            ),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(4958)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(4958)),
        ),
        (
            loss_fns.MSELoss,
            loss_fns.MSELossVR,
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8],
                out_size=1,
                bias=False,
                cov=[
                    None,
                    None,
                    bnn.params.FactorizedCovariance(),
                ],
            ),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(4958)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(4958)),
        ),
        (
            loss_fns.MSELoss,
            loss_fns.MSELossVR,
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8],
                out_size=1,
                bias=True,
                cov=[
                    None,
                    None,
                    bnn.params.FactorizedCovariance(),
                ],
            ),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(4958)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(4958)),
        ),
    ],
)
def test_equals_expected_loss(
    loss_fn, loss_fn_variance_reduced, model, input, target, reduction
):

    # TODO: temporary until all models get a way to do a forward pass through just part of the model
    # TODO: Give models in inferno.models a .representation(input) or .representation(input) function
    model_representation = bnn.Sequential(
        *(module for name, module in list(model._modules.items())[0:-2])
    )
    model = bnn.Sequential(model_representation, model[-2])

    # Evaluate loss functions
    num_samples = 10000
    loss = loss_fn(reduction=reduction)(
        model(
            input,
            sample_shape=(num_samples,),
            generator=torch.Generator().manual_seed(999),
        ),
        target,
    )
    loss_variance_reduced = loss_fn_variance_reduced(reduction=reduction)(
        model_representation(
            input,
            sample_shape=(num_samples,),
            generator=torch.Generator().manual_seed(999),
        ),
        model[-1],
        target,
    )

    npt.assert_allclose(
        loss_variance_reduced.detach().numpy(),
        loss.detach().numpy(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize(
    "loss_fn,loss_fn_variance_reduced,model,input,target",
    [
        (
            nn.MSELoss,
            loss_fns.MSELossVR,
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8, 8],
                out_size=1,
                cov=None,
            ),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(8932)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(8932)),
        ),
        (
            nn.MSELoss,
            loss_fns.MSELossVR,
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8],
                out_size=1,
                bias=False,
                cov=None,
            ),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(97)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(97)),
        ),
    ],
)
def test_equals_torch_loss_for_deterministic_models(
    loss_fn, loss_fn_variance_reduced, model, input, target, reduction
):

    # TODO: temporary until all models get a way to do a forward pass through just part of the model
    # TODO: Give models in inferno.models a .representation(input) or .representation(input) function
    model_representation = bnn.Sequential(
        *(module for name, module in list(model._modules.items())[0:-2])
    )
    model = bnn.Sequential(model_representation, model[-2])

    loss = loss_fn(reduction=reduction)(
        model(
            input,
            sample_shape=None,
        ),
        target,
    )
    loss_variance_reduced = loss_fn_variance_reduced(reduction=reduction)(
        model_representation(
            input,
            sample_shape=None,
        ),
        model[-1],
        target,
    )

    npt.assert_allclose(
        loss_variance_reduced.detach().numpy(),
        loss.detach().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
