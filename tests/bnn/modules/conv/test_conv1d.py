import math

import numpy.testing as npt
import torch
from torch import nn

from inferno import bnn
from inferno.bnn import params

import pytest


@pytest.mark.parametrize("batch_shape", [(), (1,), (4,)])
@pytest.mark.parametrize("in_channels, out_channels", [(1, 1), (1, 2)])
@pytest.mark.parametrize("in_features, kernel_size", [(1, 1), (4, 2)])
@pytest.mark.parametrize("cov", [None, bnn.params.FactorizedCovariance()])
def test_generalizes_pytorch_conv1d_layer(
    batch_shape,
    in_channels,
    out_channels,
    in_features,
    kernel_size,
    cov,
):
    """Test whether the BNN Conv1d layer generalizes the PyTorch Conv1d layer."""
    generator = torch.Generator().manual_seed(0)

    input = torch.randn(
        batch_shape + (in_channels, in_features),
        generator=generator,
    )
    conv_layer = torch.nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
    )
    conv_layer.weight = torch.nn.Parameter(conv_layer.weight)
    conv_layer.bias = torch.nn.Parameter(conv_layer.bias)

    bnn_conv_layer = bnn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        cov=cov,
    )
    bnn_conv_layer.params.weight = torch.nn.Parameter(conv_layer.weight)
    bnn_conv_layer.params.bias = torch.nn.Parameter(conv_layer.bias)

    npt.assert_allclose(
        conv_layer(input).detach().numpy(),
        bnn_conv_layer(input, sample_shape=None, generator=generator).detach().numpy(),
        atol=1e-6,
        rtol=1e-6,
    )

    if bnn_conv_layer.params.cov is not None:
        for name, param in bnn_conv_layer.params.cov.named_parameters():
            torch.nn.init.constant_(param, 0.0)

        npt.assert_allclose(
            conv_layer(input).detach().numpy(),
            bnn_conv_layer(input, sample_shape=(), generator=generator)
            .detach()
            .numpy(),
            atol=1e-6,
            rtol=1e-6,
        )


@pytest.mark.parametrize("sample_shape", [(), (1,), (2,), (3, 2)])
@pytest.mark.parametrize("batch_shape", [(), (1,), (3,)])
@pytest.mark.parametrize("in_features", [2, 5])
@pytest.mark.parametrize(
    "conv1d",
    [
        bnn.Conv1d(in_channels=3, out_channels=1, kernel_size=1),
        bnn.Conv1d(in_channels=1, out_channels=2, bias=False, kernel_size=2),
    ],
)
def test_shape(sample_shape, batch_shape, in_features, conv1d):
    """Test whether the output shape is correct."""
    generator = torch.Generator().manual_seed(0)
    input = torch.randn(
        *batch_shape, conv1d.in_channels, in_features, generator=generator
    )
    output = conv1d(input, sample_shape=sample_shape, generator=generator)

    assert output.shape == (
        *sample_shape,
        *nn.Conv1d(
            in_channels=conv1d.in_channels,
            out_channels=conv1d.out_channels,
            kernel_size=conv1d.kernel_size,
        )(input).shape,
    )


@pytest.mark.parametrize("seed", [0, 45234, 42])
def test_forward_is_deterministic_given_generator(seed):
    """Test whether the forward method is deterministic given a generator."""
    conv_layer = bnn.Conv1d(5, 3, kernel_size=2)

    input = torch.randn(5, 9, generator=torch.Generator().manual_seed(seed + 2452345))
    output1 = conv_layer(input, generator=torch.Generator().manual_seed(seed))
    output2 = conv_layer(input, generator=torch.Generator().manual_seed(seed))

    npt.assert_allclose(output1.detach().numpy(), output2.detach().numpy())


@pytest.mark.parametrize(
    "conv_layer_to_load",
    [
        bnn.Conv1d(3, 2, kernel_size=1),
        bnn.Conv1d(3, 2, kernel_size=1, cov=params.FactorizedCovariance()),
        nn.Conv1d(3, 2, kernel_size=2),
    ],
)
def test_load_from_state_dict(conv_layer_to_load):
    """Test whether the load_from_state_dict method is working for torch and inferno
    Conv1d layers."""
    state_dict = conv_layer_to_load.state_dict()
    new_conv_layer = bnn.Conv1d(
        conv_layer_to_load.in_channels,
        conv_layer_to_load.out_channels,
        kernel_size=conv_layer_to_load.kernel_size,
        cov=params.FactorizedCovariance(),
    )
    new_conv_layer.load_state_dict(
        state_dict,
        strict=hasattr(conv_layer_to_load, "params.cov"),
    )

    prefix = "params." if isinstance(conv_layer_to_load, bnn.BNNMixin) else ""

    npt.assert_allclose(
        new_conv_layer.params.weight.detach().numpy(),
        state_dict[prefix + "weight"].detach().numpy(),
    )
    npt.assert_allclose(
        new_conv_layer.params.bias.detach().numpy(),
        state_dict[prefix + "bias"].detach().numpy(),
    )

    if hasattr(conv_layer_to_load, "params.cov"):
        npt.assert_allclose(
            new_conv_layer.params.cov_params.detach().numpy(),
            state_dict[prefix + "cov_params"].detach().numpy(),
        )


def test_register_forward_hook():
    """Test whether the register_forward_hook method is working."""
    generator = torch.Generator().manual_seed(45)

    conv_layer = bnn.Conv1d(3, 2, kernel_size=1)
    test_dict = {"hook_has_fired": False}

    def hook(module, input, output):
        test_dict["hook_has_fired"] = True

    conv_layer.register_forward_hook(hook)
    conv_layer(torch.randn(5, 3, 9, generator=generator))

    assert test_dict["hook_has_fired"]


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
def test_if_bias_is_none_cov_bias_is_none(cov):
    """Test whether if the bias is None the covariance doesn't have bias parameters."""
    linear_layer = bnn.Conv1d(5, 3, kernel_size=5, bias=False, cov=cov)

    for name, param in linear_layer.params.cov.named_parameters():
        if "bias" in name:
            assert param is None


@pytest.mark.parametrize(
    "parametrization",
    [
        params.Standard(),
        params.NeuralTangent(),
        params.MaximalUpdate(),
    ],
    ids=lambda x: x.__class__.__name__,
)
@pytest.mark.parametrize("layer_type", ["input", "hidden", "output"], ids=lambda x: x)
def test_weight_initialization(parametrization, layer_type):
    """Test whether the initialization of the weights is correct."""
    torch.manual_seed(2452345)

    with torch.no_grad():
        linear_layer = bnn.Conv1d(
            300,
            300,
            kernel_size=2,
            layer_type=layer_type,
            parametrization=parametrization,
            cov=None,
        )

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
            linear_layer.params.weight
        )

        npt.assert_allclose(
            linear_layer.weight.mean(),
            0.0,
            atol=1e-2,
            rtol=1e-2,
        )

        npt.assert_allclose(
            linear_layer.weight.std(),
            parametrization.weight_init_scale(
                fan_in=fan_in, fan_out=fan_out, layer_type=layer_type
            ),
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.parametrize(
    "parametrization",
    [
        params.Standard(),
        params.NeuralTangent(),
        params.MaximalUpdate(),
    ],
    ids=lambda x: x.__class__.__name__,
)
@pytest.mark.parametrize("layer_type", ["input", "hidden", "output"], ids=lambda x: x)
def test_bias_initialization(parametrization, layer_type):
    """Test whether the initialization of the bias is correct."""
    torch.manual_seed(2452345)

    with torch.no_grad():
        linear_layer = bnn.Conv1d(
            300,
            300,
            kernel_size=2,
            bias=True,
            layer_type=layer_type,
            parametrization=parametrization,
            cov=None,
        )

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
            linear_layer.params.weight
        )

        npt.assert_allclose(
            linear_layer.bias.mean(),
            0.0,
            atol=1e-1,
            rtol=1e-1,
        )

        npt.assert_allclose(
            linear_layer.bias.std(),
            parametrization.bias_init_scale(
                fan_in=fan_in, fan_out=fan_out, layer_type=layer_type
            ),
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.parametrize(
    "parametrization",
    [
        params.Standard(),
        params.NeuralTangent(),
        params.MaximalUpdate(),
    ],
    ids=lambda x: x.__class__.__name__,
)
@pytest.mark.parametrize("layer_type", ["input", "hidden", "output"], ids=lambda x: x)
@pytest.mark.parametrize(
    "layer_width,cov",
    [
        (30, params.FactorizedCovariance()),
        (100, params.LowRankCovariance(5)),
        (100, params.KroneckerCovariance()),
        (100, params.DiagonalCovariance()),
    ],
    ids=lambda x: (
        x.__class__.__name__
        if isinstance(x, params.FactorizedCovariance)
        else f"width_{x}"
    ),
)
def test_covariance_parameter_initialization(
    parametrization, layer_type, layer_width, cov
):
    """Test whether the initialization of the covariance parameters is correct."""
    torch.manual_seed(2452345)

    with torch.no_grad():
        linear_layer = bnn.Conv1d(
            layer_width,
            layer_width,
            kernel_size=2,
            bias=True,
            layer_type=layer_type,
            parametrization=parametrization,
            cov=cov,
        )

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
            linear_layer.params.weight
        )

        if isinstance(cov, params.DiagonalCovariance):
            covariance_specific_scaling_factor = 1.0
        elif isinstance(cov, params.KroneckerCovariance):
            covariance_specific_scaling_factor = 1 / math.sqrt(cov.input_rank)
        elif isinstance(cov, params.LowRankCovariance):
            covariance_specific_scaling_factor = 1 / math.sqrt(cov.rank)
        elif isinstance(cov, params.FactorizedCovariance):
            covariance_specific_scaling_factor = 1 / math.sqrt(cov.rank)
        else:
            raise NotImplementedError()

        for name, param in linear_layer.params.cov.named_parameters():
            if param is not None:
                npt.assert_allclose(
                    param.mean(),
                    0.0,
                    atol=1e-1,
                    rtol=1e-1,
                )
            if "weight" in name:
                npt.assert_allclose(
                    param.std(),
                    parametrization.weight_init_scale(
                        fan_in=fan_in, fan_out=fan_out, layer_type=layer_type
                    )
                    * covariance_specific_scaling_factor,
                    atol=1e-1,
                    rtol=1e-1,
                )
            elif "bias" in name:
                npt.assert_allclose(
                    param.std(),
                    parametrization.bias_init_scale(
                        fan_in=fan_in, fan_out=fan_out, layer_type=layer_type
                    )
                    * covariance_specific_scaling_factor,
                    atol=1e-1,
                    rtol=1e-1,
                )
