import copy

import numpy.testing as npt
import torch
from torch import nn
import torchvision

import inferno
from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "inferno_mlp,torchvision_mlp",
    [
        (
            inferno.models.MLP(
                in_size=100,
                hidden_sizes=[64, 64],
                out_size=10,
                norm_layer=None,
                activation_layer=nn.ReLU,
                dropout=0.0,
                bias=True,
                cov=None,
            ),
            torchvision.ops.MLP(
                in_channels=100,
                hidden_channels=[64, 64, 10],
                norm_layer=None,
                activation_layer=nn.ReLU,
                inplace=None,
                bias=True,
            ),
        ),
        (
            inferno.models.MLP(
                in_size=100,
                hidden_sizes=[64, 64],
                out_size=10,
                norm_layer=nn.LayerNorm,
                activation_layer=nn.SiLU,
                dropout=0.5,
                bias=True,
                cov=None,
            ),
            torchvision.ops.MLP(
                in_channels=100,
                hidden_channels=[64, 64, 10],
                norm_layer=nn.LayerNorm,
                activation_layer=nn.SiLU,
                dropout=0.5,
                inplace=None,
                bias=True,
            ),
        ),
    ],
)
def test_same_as_torchvision_mlp(inferno_mlp, torchvision_mlp):
    """Test whether the implementation matches the one of torchvision if no covariance is used."""
    torch.manual_seed(0)

    # Load weights from torchvision model
    state_dict = torchvision_mlp.state_dict()
    inferno_mlp.load_state_dict(state_dict, strict=True)

    # Create random input
    input = torch.randn((2, 100))

    # Forward pass through both models
    inferno_mlp.eval()
    torchvision_mlp.eval()
    with torch.no_grad():
        inferno_output = inferno_mlp(input, sample_shape=(2,))[
            0
        ]  # Draw multiple samples to check batch compatibility

        torchvision_output = torchvision_mlp(input)

    # Compare the outputs
    npt.assert_allclose(
        inferno_output.detach().numpy(),
        torchvision_output.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "model",
    [
        inferno.models.MLP(
            in_size=10,
            hidden_sizes=[32, 32],
            out_size=10,
            norm_layer=nn.LayerNorm,
            activation_layer=nn.SiLU,
            dropout=0.1,
            bias=True,
            cov=params.KroneckerCovariance(),
        ),
        inferno.models.MLP(
            in_size=10,
            hidden_sizes=[32, 32],
            out_size=10,
            norm_layer=nn.LayerNorm,
            activation_layer=nn.SiLU,
            dropout=0.1,
            bias=True,
            cov=[None, None, params.FactorizedCovariance()],
        ),
    ],
)
def test_sample_shape_none_corresponds_to_forward_pass_with_mean_params(model):

    deterministic_model = copy.deepcopy(model)
    for layer in deterministic_model:
        if hasattr(layer, "params"):
            layer.params.cov = None

    input = torch.randn(3, 10, generator=torch.Generator().manual_seed(4236))

    model.eval()
    deterministic_model.eval()

    npt.assert_allclose(
        deterministic_model(input).detach().numpy(),
        model(input, sample_shape=None).detach().numpy(),
    )


@pytest.mark.parametrize(
    "model",
    [
        inferno.models.MLP(
            in_size=10,
            hidden_sizes=[32, 32],
            out_size=10,
            norm_layer=nn.LayerNorm,
            activation_layer=nn.SiLU,
            dropout=0.1,
            bias=True,
            cov=params.KroneckerCovariance(),
        ),
        inferno.models.MLP(
            in_size=10,
            hidden_sizes=[32, 32],
            out_size=10,
            norm_layer=nn.LayerNorm,
            activation_layer=nn.SiLU,
            dropout=0.1,
            bias=True,
            cov=[None, None, params.FactorizedCovariance()],
        ),
        inferno.models.MLP(
            in_size=10,
            hidden_sizes=[32, 32],
            out_size=(),
            norm_layer=nn.LayerNorm,
            activation_layer=nn.SiLU,
            bias=True,
            cov=[None, None, params.FactorizedCovariance()],
        ),
        inferno.models.MLP(
            in_size=10,
            hidden_sizes=[32, 32],
            out_size=(2, 3),
            norm_layer=nn.LayerNorm,
            activation_layer=nn.SiLU,
            bias=True,
            cov=[None, None, params.FactorizedCovariance()],
        ),
    ],
)
def test_draw_samples(model):
    """Test whether the model can draw samples."""
    torch.manual_seed(0)

    in_size = 10

    # Create random input
    batch_size = 8
    input = torch.randn((batch_size, in_size))

    # Forward pass through the model
    model.eval()
    sample_shape = (10,)
    with torch.no_grad():
        output = model(input, sample_shape=sample_shape)

    # Check the shape of the output
    out_size = (
        model.out_size if isinstance(model.out_size, tuple) else (model.out_size,)
    )
    assert output.shape == (*sample_shape, batch_size, *out_size)
