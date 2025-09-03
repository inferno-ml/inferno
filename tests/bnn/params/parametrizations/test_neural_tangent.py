import math

from numpy import testing as npt
import torch
from torch import nn

from inferno import bnn
from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "layer_class,kwargs",
    [
        (bnn.Linear, {}),
        (bnn.Conv1d, {"kernel_size": 1}),
        (bnn.Conv2d, {"kernel_size": 1}),
        (bnn.Conv3d, {"kernel_size": 1}),
    ],
)
@pytest.mark.parametrize("layer_type", ["input", "output", "hidden"])
@pytest.mark.parametrize(
    "cov", [params.FactorizedCovariance(), params.LowRankCovariance(500)]
)
def test_parametrization_initializes_parameters_correctly(
    layer_class,
    kwargs,
    layer_type,
    cov,
):
    torch.manual_seed(348)

    layer = layer_class(
        50,
        20,
        bias=True,
        layer_type=layer_type,
        cov=cov,
        parametrization=params.NeuralTangent(),
        **kwargs,
    )

    weight_init_scale = 1
    bias_init_scale = 1
    numel_mean_parameters = sum(
        p.numel() for name, p in layer.named_parameters() if "cov." not in name
    )
    cov_params_init_scale = weight_init_scale / math.sqrt(numel_mean_parameters)

    npt.assert_allclose(
        layer.weight.detach().numpy().std(),
        weight_init_scale,
        atol=1e-1,
        rtol=1e-1,
    )

    npt.assert_allclose(
        layer.bias.detach().numpy().std(),
        bias_init_scale,
        atol=1e-1,
        rtol=1e-1,
    )

    for name, param in layer.params.cov.named_parameters():
        npt.assert_allclose(
            param.detach().numpy().std(),
            cov_params_init_scale,
            atol=1e-1,
            rtol=1e-1,
        )
