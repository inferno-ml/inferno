import copy

import numpy.testing as npt
import torch
from torch import nn
import torchvision

import inferno
from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "inferno_resnet,torchvision_resnet",
    [
        (
            inferno.models.ResNet(
                block=inferno.models.resnet.BasicBlock,
                num_blocks_per_layer=[2, 2, 2, 2],
                out_size=1000,
                cov=None,
            ),
            torchvision.models.ResNet(
                block=torchvision.models.resnet.BasicBlock,
                layers=[2, 2, 2, 2],
                norm_layer=lambda c: nn.GroupNorm(num_groups=32, num_channels=c),
            ),
        ),
        (
            inferno.models.ResNet(
                block=inferno.models.resnet.Bottleneck,
                num_blocks_per_layer=[2, 2, 2, 2],
                out_size=1000,
                cov=None,
            ),
            torchvision.models.ResNet(
                block=torchvision.models.resnet.Bottleneck,
                layers=[2, 2, 2, 2],
                norm_layer=lambda c: nn.GroupNorm(num_groups=32, num_channels=c),
            ),
        ),
    ],
)
def test_same_as_torchvision_resnet(inferno_resnet, torchvision_resnet):
    """Test whether the implementation matches the one of torchvision if no covariance is used."""
    torch.manual_seed(0)

    # Load weights from torchvision model
    state_dict = torchvision_resnet.state_dict()
    inferno_resnet.load_state_dict(state_dict, strict=True)

    # Create random input
    input = torch.randn((1, 3, 224, 224))

    # Forward pass through both models
    inferno_resnet.eval()
    torchvision_resnet.eval()
    with torch.no_grad():
        inferno_output = inferno_resnet(input, sample_shape=(2,))[
            0
        ]  # Draw multiple samples to check batch compatibility
        torchvision_output = torchvision_resnet(input)

    # Compare the outputs
    npt.assert_allclose(
        inferno_output.detach().numpy(),
        torchvision_output.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "resnet_type,weights,out_size,architecture,cov",
    [
        (
            inferno.models.ResNet18,
            torchvision.models.ResNet18_Weights.DEFAULT,
            10,
            "cifar",
            params.LowRankCovariance(10),
        ),
        (
            inferno.models.ResNet18,
            torchvision.models.ResNet18_Weights.DEFAULT,
            100,
            "cifar",
            params.LowRankCovariance(10),
        ),
        (
            inferno.models.ResNet18,
            torchvision.models.ResNet18_Weights.DEFAULT,
            200,
            "imagenet",
            params.LowRankCovariance(10),
        ),
    ],
)
def test_sample_shape_none_corresponds_to_forward_pass_with_mean_params(
    resnet_type, weights, out_size, architecture, cov
):
    deterministic_model = resnet_type.from_pretrained_weights(
        weights=weights,
        out_size=out_size,
        architecture=architecture,
        cov=None,
    )
    model = resnet_type(
        out_size=out_size,
        architecture=architecture,
        cov=cov,
    )

    model.load_state_dict(deterministic_model.state_dict(), strict=False)

    if architecture == "cifar":
        input_shape = (3, 32, 32)
    else:
        input_shape = (3, 224, 224)

    input = torch.randn(
        (8,) + input_shape, generator=torch.Generator().manual_seed(543)
    )

    model.eval()
    deterministic_model.eval()

    npt.assert_allclose(
        deterministic_model(input).detach().numpy(),
        model(input, sample_shape=None).detach().numpy(),
    )


def test_draw_samples():
    """Test whether the model can draw samples."""
    torch.manual_seed(0)

    # Create a ResNet model
    model = inferno.models.ResNet(
        block=inferno.models.resnet.BasicBlock,
        num_blocks_per_layer=[2, 2, 2, 2],
        cov=params.KroneckerCovariance(),
        out_size=1000,
    )

    # Create random input
    batch_size = 8
    input = torch.randn((batch_size, 3, 224, 224))

    # Forward pass through the model
    model.eval()
    sample_shape = (10,)
    with torch.no_grad():
        output = model(input, sample_shape=sample_shape)

    # Check the shape of the output
    assert output.shape == (*sample_shape, batch_size, 1000)


def test_batch_norm_raises_value_error():
    """Test whether the model raises a ValueError when batch normalization is used with a non-batch dimension."""
    torch.manual_seed(0)

    with pytest.raises(ValueError):
        # Create a ResNet model with batch normalization
        model = inferno.models.ResNet(
            block=inferno.models.resnet.BasicBlock,
            num_blocks_per_layer=[2, 2, 2, 2],
            norm_layer=nn.BatchNorm2d,
            out_size=1000,
        )


@pytest.mark.parametrize(
    "resnet_type,weights,out_size,architecture,cov,freeze",
    [
        (
            inferno.models.ResNet18,
            torchvision.models.ResNet18_Weights.DEFAULT,
            10,
            "cifar",
            None,
            False,
        ),
        (
            inferno.models.ResNet18,
            torchvision.models.ResNet18_Weights.DEFAULT,
            100,
            "cifar",
            params.LowRankCovariance(100),
            True,
        ),
        (
            inferno.models.ResNet18,
            torchvision.models.ResNet18_Weights.DEFAULT,
            200,
            "imagenet",
            None,
            False,
        ),
        (
            inferno.models.ResNet34,
            torchvision.models.ResNet34_Weights.DEFAULT,
            1000,
            "imagenet",
            params.LowRankCovariance(100),
            True,
        ),
    ],
)
def test_from_pretrained_weights(
    resnet_type, weights, out_size, architecture, cov, freeze
):
    """Test whether the model can be loaded with pretrained weights."""
    torch.manual_seed(0)

    pretrained_model = resnet_type.from_pretrained_weights(
        weights=weights,
        out_size=out_size,
        architecture=architecture,
        cov=cov,
        freeze=freeze,
    )

    pretrained_weights_state_dict = weights.get_state_dict()

    # Check whether weights are loaded correctly
    npt.assert_allclose(
        pretrained_model.state_dict()["layer1.1.conv1.params.weight"].detach().numpy(),
        pretrained_weights_state_dict["layer1.1.conv1.weight"].detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    npt.assert_allclose(
        pretrained_model.state_dict()["layer2.1.conv1.params.weight"].detach().numpy(),
        pretrained_weights_state_dict["layer2.1.conv1.weight"].detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    # Check if freezing the weights works
    if freeze:
        for name, param in pretrained_model.named_parameters():
            if name.replace(".params", "") in pretrained_weights_state_dict:
                if name in [
                    "conv1.params.weight",
                    "fc.params.weight",
                    "fc.params.bias",
                ]:
                    # First and last layer may be trainable
                    continue

                assert not param.requires_grad, name
