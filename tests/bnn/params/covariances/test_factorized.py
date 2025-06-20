from numpy import testing as npt
import torch

from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "covariance,mean_parameters",
    [
        (
            params.FactorizedCovariance(),
            {"weight": torch.randn(4, 2), "bias": torch.randn(4)},
        ),
        (
            params.FactorizedCovariance(),
            {"weight": torch.randn(1, 2, 3, 2), "bias": torch.randn((1, 3, 2))},
        ),
        (
            params.FactorizedCovariance(),
            {"weight": torch.randn(1, 2), "bias": torch.randn((1,))},
        ),
        (
            params.FactorizedCovariance(),
            {
                "weight": torch.randn(
                    1,
                )
            },
        ),
    ],
)
def test_factor_matmul(covariance, mean_parameters):
    """Test whether the multiplication with the covariance factor is correct."""

    torch.manual_seed(2435)

    numel_parameters = sum(
        [tens.numel() for tens in mean_parameters.values() if tens is not None]
    )

    covariance.initialize_parameters(mean_parameters)
    covariance.reset_parameters({name: 1.0 for name in mean_parameters.keys()})

    matmul_result_dict = covariance.factor_matmul(
        torch.eye(covariance.rank), additive_constant=torch.zeros(numel_parameters)
    )

    for name, tens in matmul_result_dict.items():
        npt.assert_allclose(
            torch.movedim(tens, 0, -1)
            .detach()
            .numpy(),  # Rank dimension is in front for factor_matmul
            covariance.factor[name].detach().numpy(),
        )
