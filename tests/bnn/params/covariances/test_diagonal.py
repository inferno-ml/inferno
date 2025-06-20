from numpy import testing as npt
import torch

from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "mean_parameters",
    [
        {"weight": torch.randn(4, 2), "bias": torch.randn(4)},
        {"weight": torch.randn(1, 2), "bias": torch.randn((1,))},
        {"weight": torch.randn(1, 2, 3, 2), "bias": torch.randn((1, 3, 2))},
        {
            "weight": torch.randn(
                1,
            )
        },
    ],
)
def test_factor_matmul_low_rank_covariance(mean_parameters):
    """Test whether the multiplication with the covariance factor is correct."""
    torch.manual_seed(2345)

    covariance = params.DiagonalCovariance()
    numel_mean_parameters = sum(
        [tens.numel() for tens in mean_parameters.values() if tens is not None]
    )

    covariance.initialize_parameters(mean_parameters)
    covariance.reset_parameters({name: 1.0 for name in mean_parameters.keys()})

    input = torch.randn((3, 2, 1, numel_mean_parameters))
    matmul_result_dict = covariance.factor_matmul(
        input,
        additive_constant=torch.zeros(numel_mean_parameters),
    )

    stacked_scale_params = torch.hstack(
        [
            scale_param.view(-1)
            for scale_param in covariance.scale.values()
            if scale_param is not None
        ]
    )

    npt.assert_allclose(
        input * stacked_scale_params.detach().numpy(),
        torch.concatenate(
            [
                scale_param.view(*input.shape[:-1], -1)
                for scale_param in matmul_result_dict.values()
            ],
            dim=-1,
        )
        .detach()
        .numpy(),
    )
