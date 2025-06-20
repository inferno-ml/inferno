import copy

import numpy as np
from numpy import testing as npt
import torch

from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "mean",
    [
        {"weight": torch.randn(4, 2), "bias": torch.randn(4)},
        {"weight": torch.randn(4, 2), "bias": None},
        {"weight": torch.randn(1, 1)},
        {"weight": torch.randn(3, 2, 1, 2), "bias": torch.randn((3,))},
    ],
)
@pytest.mark.parametrize(
    "cov",
    [
        params.FactorizedCovariance(),
        params.DiagonalCovariance(),
        params.LowRankCovariance(rank=2),
        params.KroneckerCovariance(input_rank=2, output_rank=3),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_sample(mean, cov):
    """Test sampling from a Gaussian parameter."""
    torch.random.manual_seed(463)

    # Initialize the Gaussian parameter
    cov = copy.deepcopy(cov)
    param = params.GaussianParameter(mean, cov)
    param.cov.reset_parameters()

    # Sample
    sample_shape = torch.Size((100000,))
    parameter_samples = param.sample(sample_shape)

    for name, tens in parameter_samples.items():
        if param[name] is None:
            assert tens is None
        else:
            # Check that the shape is correct
            assert tens.shape == sample_shape + param[name].shape

            # Check that the mean is correct
            npt.assert_allclose(
                tens.mean(dim=tuple(np.arange(len(sample_shape)))).detach().numpy(),
                param[name].detach().numpy(),
                rtol=1e-1,
                atol=1e-1,
            )

    # Check that the covariance is correct
    npt.assert_allclose(
        torch.hstack(
            [tens.view(*sample_shape, -1) for tens in parameter_samples.values()]
        )
        .mT.cov()
        .detach()
        .numpy(),
        param.cov.to_dense().detach().numpy(),
        rtol=1e-1,
        atol=1e-1,
    )
