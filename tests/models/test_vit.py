from collections.abc import MutableMapping
import copy
import itertools

import torch
from torch import nn, testing
import torchvision

import inferno
from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "inferno_vit,torchvision_vit",
    [
        (
            inferno.models.VisionTransformer(
                in_size=32,
                patch_size=2,
                num_layers=2,
                num_heads=2,
                hidden_dim=10,
                mlp_dim=10,
                out_size=5,
                representation_size=7,
                cov=None,
            ),
            torchvision.models.VisionTransformer(
                image_size=32,
                patch_size=2,
                num_layers=2,
                num_heads=2,
                hidden_dim=10,
                mlp_dim=10,
                num_classes=5,
                representation_size=7,
            ),
        ),
    ],
)
def test_same_as_torchvision_vit(inferno_vit, torchvision_vit):
    """Test whether the implementation matches the one of torchvision if no covariance is used."""
    torch.manual_seed(0)

    # Load weights from torchvision model
    state_dict = torchvision_vit.state_dict()
    inferno_vit.load_state_dict(state_dict, strict=True)

    # Create random input
    input = torch.randn((1, 3, 32, 32))

    # Forward pass through both models
    inferno_vit.eval()
    torchvision_vit.eval()
    with torch.no_grad():
        inferno_output = inferno_vit(input, sample_shape=(2,))[
            0
        ]  # Draw multiple samples to check batch compatibility
        # inferno_output = inferno_vit(input)
        torchvision_output = torchvision_vit(input)

    # Compare the outputs
    testing.assert_close(
        inferno_output,
        torchvision_output,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "vit_type,out_size,architecture,cov",
    [
        # (
        #    inferno.models.ViT_B_16,
        #    10,
        #    "cifar",
        #    params.LowRankCovariance(10),
        # ),
        (
            inferno.models.ViT_B_16,
            1000,
            "imagenet",
            params.LowRankCovariance(2),
        ),
        # (
        #    inferno.models.ResNet18,v
        #    100,
        #    "cifar",
        #    params.LowRankCovariance(10),
        # ),
        # (
        #    inferno.models.ResNet18,
        #    200,
        #    "imagenet",
        #    params.LowRankCovariance(10),
        # ),
    ],
)
def test_sample_shape_none_corresponds_to_forward_pass_with_mean_params(
    vit_type, out_size, architecture, cov
):
    deterministic_model = vit_type.from_pretrained_weights(
        out_size=out_size,
        architecture=architecture,
        cov=None,
    )
    model = vit_type(
        in_size=32 if architecture == "cifar" else 224,
        out_size=out_size,
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

    testing.assert_close(
        deterministic_model(input),
        model(input, sample_shape=None),
    )


def test_draw_samples():
    """Test whether the model can draw samples."""
    torch.manual_seed(0)

    # Create a VisionTransformer model
    model = inferno.models.VisionTransformer(
        in_size=224,
        patch_size=16,
        num_layers=2,
        num_heads=2,
        hidden_dim=128,
        mlp_dim=10,
        out_size=1000,
        cov=params.LowRankCovariance(10),
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
        # Create a VisionTransformer model with batch normalization
        model = inferno.models.VisionTransformer(
            in_size=224,
            patch_size=16,
            num_layers=2,
            num_heads=2,
            hidden_dim=128,
            mlp_dim=10,
            out_size=1000,
            norm_layer=nn.BatchNorm2d,
        )


@pytest.mark.parametrize(
    "vit_type,weights,out_size,architecture,cov,freeze",
    [
        (
            inferno.models.ViT_B_16,
            torchvision.models.ViT_B_16_Weights.DEFAULT,
            10,
            "cifar",
            None,
            False,
        ),
        (
            inferno.models.ViT_B_16,
            torchvision.models.ViT_B_16_Weights.DEFAULT,
            100,
            "cifar",
            params.LowRankCovariance(10),
            True,
        ),
        # (
        #    inferno.models.ResNet18,
        #    torchvision.models.ResNet18_Weights.DEFAULT,
        #    200,
        #    "imagenet",
        #    None,
        #    False,
        # ),
        # (
        #    inferno.models.ResNet34,
        #    torchvision.models.ResNet34_Weights.DEFAULT,
        #    1000,
        #    "imagenet",
        #    params.LowRankCovariance(100),
        #    True,
        # ),
    ],
)
def test_from_pretrained_weights(
    vit_type, weights, out_size, architecture, cov, freeze
):
    """Test whether the model can be loaded with pretrained weights."""
    torch.manual_seed(0)

    pretrained_model = vit_type.from_pretrained_weights(
        weights=weights,
        out_size=out_size,
        architecture=architecture,
        cov=cov,
        freeze=freeze,
    )

    pretrained_weights_state_dict = weights.get_state_dict()

    # Check whether weights are loaded correctly
    testing.assert_close(
        pretrained_model.state_dict()[
            "encoder.layers.encoder_layer_0.self_attention.out_proj.params.weight"
        ]
        .detach()
        .numpy(),
        pretrained_weights_state_dict[
            "encoder.layers.encoder_layer_0.self_attention.out_proj.weight"
        ]
        .detach()
        .numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    testing.assert_close(
        pretrained_model.state_dict()[
            "encoder.layers.encoder_layer_1.self_attention.out_proj.params.weight"
        ]
        .detach()
        .numpy(),
        pretrained_weights_state_dict[
            "encoder.layers.encoder_layer_1.self_attention.out_proj.weight"
        ]
        .detach()
        .numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    # Check if freezing the weights works
    if freeze:
        for name, param in pretrained_model.named_parameters():
            if name.replace(".params", "") in pretrained_weights_state_dict:
                if name in [
                    "conv_proj.params.weight",
                    "conv_proj.params.bias",
                    "encoder.pos_embedding",
                    "heads.head.params.weight",
                    "heads.head.params.bias",
                ]:
                    # First and last layer may be trainable
                    continue

                assert not param.requires_grad, name


@pytest.mark.parametrize(
    "cov,cov_is_correct",
    [
        (None, True),
        (params.DiagonalCovariance(), True),
        (
            {
                "conv_proj": None,
                "encoder": params.LowRankCovariance(2),
                "heads.pre_logits": params.DiagonalCovariance(),
                "heads.head": params.KroneckerCovariance(),
            },
            True,
        ),
        (
            {
                "conv_proj": params.DiagonalCovariance(),
                "encoder": {
                    "layers.encoder_layer_0": params.LowRankCovariance(2),
                    "layers.encoder_layer_1": params.DiagonalCovariance(),
                },
                "heads.pre_logits": params.DiagonalCovariance(),
            },
            True,
        ),
        (
            {
                "conv_proj": params.DiagonalCovariance(),
                "encoder": {
                    "layers.encoder_layer_0": params.LowRankCovariance(2),
                    "layers.encoder_layer_1": {
                        "self_attention": params.DiagonalCovariance()
                    },
                },
                "heads.pre_logits": params.KroneckerCovariance(),
                "heads.head": params.DiagonalCovariance(),
            },
            True,
        ),
        (
            {
                "conv_proj": params.DiagonalCovariance(),
                "encoder": {
                    "layers.encoder_layer_0": params.LowRankCovariance(2),
                    "layers.encoder_layer_1": {
                        "self_attention": {
                            "q": params.KroneckerCovariance(),
                            "k": params.DiagonalCovariance(),
                            "v": params.LowRankCovariance(2),
                            "out": params.DiagonalCovariance(),
                        }
                    },
                },
                "heads.pre_logits": params.KroneckerCovariance(),
                "heads.head": params.DiagonalCovariance(),
            },
            True,
        ),
        (
            {
                "conv_proj": None,
                "encoder": params.LowRankCovariance(2),
                "pre_logits": params.DiagonalCovariance(),
                "head": params.KroneckerCovariance(),
            },
            False,
        ),
    ],
)
def test_covariance_spec(cov, cov_is_correct):
    """Test whether the covariance can be specified in various ways."""
    torch.manual_seed(0)

    if not cov_is_correct:
        with pytest.raises(ValueError):
            model = inferno.models.VisionTransformer(
                in_size=32,
                patch_size=2,
                num_layers=2,
                num_heads=2,
                hidden_dim=10,
                mlp_dim=10,
                out_size=5,
                representation_size=7,
                cov=cov,
            )
    else:
        model = inferno.models.VisionTransformer(
            in_size=32,
            patch_size=2,
            num_layers=2,
            num_heads=2,
            hidden_dim=10,
            mlp_dim=10,
            out_size=5,
            representation_size=7,
            cov=cov,
        )

        # case 1: no covariance
        if cov is None:
            for name, module in model.named_modules():
                # decide if it's a covariance based on name
                name_postfix = ".".join(name.split(".")[-2:])
                if name_postfix == "params.cov":
                    assert module is None

        # case 2: constant type
        elif isinstance(cov, params.FactorizedCovariance):
            for name, module in model.named_modules():
                # decide if it's a covariance based on name
                name_postfix = ".".join(name.split(".")[-2:])
                if name_postfix == "params.cov":
                    assert isinstance(module, type(cov))

        # case 3: dictionary, possibly nested
        elif isinstance(cov, dict):

            # helper function to flatten the tree of covariances
            def flatten_dict(dictionary, parent_key=""):
                items = []
                for key, value in dictionary.items():
                    new_key = parent_key + "." + key if parent_key else key
                    if isinstance(value, MutableMapping):
                        items.extend(flatten_dict(value, new_key).items())
                    else:
                        items.append((new_key, value))
                return dict(items)

            _cov_flat = flatten_dict(cov)

            # hack to fix naming inconsistency in attention
            # e.g., model name is "q_proj" but covariance must be specified as "q"
            cov_flat = {}
            for key in _cov_flat.keys():
                for old, new in [
                    ("self_attention.q", "self_attention.q_proj"),
                    ("self_attention.k", "self_attention.k_proj"),
                    ("self_attention.v", "self_attention.v_proj"),
                    ("self_attention.out", "self_attention.out_proj"),
                ]:
                    new_key = key.replace(old, new, 1)
                    cov_flat[new_key] = _cov_flat[key]

            for name, module in model.named_modules():
                # decide if it's a covariance based on name
                name_postfix = ".".join(name.split(".")[-2:])
                if name_postfix == "params.cov":
                    name_prefix = ".".join(name.split(".")[:-2])

                    # keep looking until you find a match (ie, covariance could be specified by parent module)
                    name_prefixes = list(
                        itertools.accumulate(
                            name_prefix.split("."), lambda x, y: x + "." + y
                        )
                    )

                    found_match = False
                    for name_prefix in reversed(name_prefixes):

                        # look up specified covariance type
                        if name_prefix in cov_flat.keys():
                            spec_cov_type = type(cov_flat[name_prefix])
                            if spec_cov_type is None:
                                assert module is None
                            else:
                                assert isinstance(module, spec_cov_type)
                            found_match = True
                            break

                    if not found_match:
                        assert module is None
