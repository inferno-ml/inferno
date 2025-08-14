import copy

import numpy.testing as npt
import torch
from torch import nn

from inferno import bnn, models
from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "base_module",
    [
        nn.Linear(5, 3),
        bnn.Linear(5, 6),
        bnn.Sequential(
            bnn.Linear(5, 3, cov=params.FactorizedCovariance()),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.Softmax(dim=-1),
        ),
    ],
)
@pytest.mark.parametrize("ensemble_size", [1, 3, 10])
@pytest.mark.parametrize("sample_shape", [(), (1,), (3, 2)])
@pytest.mark.parametrize("batch_shape", [(1,), (4,)])
def test_shape(base_module, ensemble_size, sample_shape, batch_shape):
    """Test whether the output shape of the ensemble is correct."""
    generator = torch.Generator().manual_seed(0)

    ensemble = models.Ensemble(
        members=[copy.deepcopy(base_module) for _ in range(ensemble_size)],
    )
    if isinstance(base_module, bnn.Sequential):
        in_features = base_module[0].in_features
    elif isinstance(base_module, (bnn.Linear, nn.Linear)):
        in_features = base_module.in_features
    else:
        raise ValueError()

    input = torch.randn((*batch_shape, in_features), generator=generator)
    output = ensemble(input, sample_shape=sample_shape, generator=generator)

    if isinstance(base_module, bnn.BNNMixin):
        kwargs = {"generator": generator}
    else:
        kwargs = {}
        sample_shape = ()

    assert output.shape == (
        ensemble_size,
        *sample_shape,
        *batch_shape,
        base_module(torch.randn((in_features,), generator=generator), **kwargs).shape[
            -1
        ],
    )


@pytest.mark.parametrize(
    "base_module",
    [
        nn.Linear(4, 1),
        bnn.Linear(4, 1, cov=params.FactorizedCovariance()),
        bnn.Sequential(
            bnn.Linear(4, 3, cov=params.LowRankCovariance(2)),
            nn.ReLU(),
            bnn.Linear(3, 1, cov=params.FactorizedCovariance()),
        ),
    ],
)
@pytest.mark.parametrize("seed", [0, 45234, 42])
def test_forward_is_deterministic_given_generator(base_module, seed):
    """Test whether the forward method is deterministic given a generator."""
    ensemble = models.Ensemble(
        members=[copy.deepcopy(base_module) for _ in range(10)],
    )

    input = torch.randn(3, 4, generator=torch.Generator().manual_seed(seed + 2452345))
    output1 = ensemble(input, generator=torch.Generator().manual_seed(seed))
    output2 = ensemble(input, generator=torch.Generator().manual_seed(seed))

    npt.assert_allclose(output1.detach().numpy(), output2.detach().numpy())


@pytest.mark.parametrize(
    "members,parametrization",
    [
        (
            [
                bnn.Linear(
                    5,
                    3,
                    cov=params.FactorizedCovariance(),
                    parametrization=params.Standard(),
                )
            ]
            * 5,
            params.Standard,
        ),
        (
            [
                bnn.Linear(
                    5,
                    3,
                    cov=params.FactorizedCovariance(),
                    parametrization=params.MaximalUpdate(),
                )
            ]
            * 5,
            params.MaximalUpdate,
        ),
        ([nn.Linear(5, 3)] * 5, params.Standard),
    ],
)
def test_parametrization(members, parametrization):
    """Test whether the parametrization of the ensemble is correct."""
    ensemble = models.Ensemble(members)

    assert isinstance(ensemble.parametrization, parametrization)

    for member in ensemble.members:
        if hasattr(member, "parametrization"):
            # Check that the parametrization of the member is the same as the ensemble
            # (if it has one)
            assert isinstance(member.parametrization, parametrization)
