"""Vision Transformers.

This implementation largely follows
[``torchvision.models.vision_transformer``](https://github.com/pytorch/vision/blob/1e53952f57462e4c28103835cf1f9e504dbea84b/torchvision/models/vision_transformer.py#L536).
"""

from __future__ import annotations

from collections import OrderedDict
import copy
from functools import partial
import math
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple, Optional

import torch
import torch.nn as nn
import torchvision
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.utils import _log_api_usage_once

from . import MLP
from .. import bnn
from ..bnn import params
from ..bnn.modules.bnn_mixin import (
    parameters_and_lrs_of_torch_module,
    reset_parameters_of_torch_module,
)

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(
        self,
        in_dim: int,
        mlp_dim: int,
        dropout: float,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: params.FactorizedCovariance | None = None,
    ):
        super().__init__(
            in_dim,
            [mlp_dim],
            in_dim,
            activation_layer=nn.GELU,
            inplace=None,
            dropout=dropout,
            parametrization=parametrization,
            cov=cov,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(bnn.BNNMixin, nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: (
            params.FactorizedCovariance | dict[params.FactorizedCovariance] | None
        ) = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        cov = check_cov(cov, ["self_attention", "mlp"])

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = bnn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=attention_dropout,
            parametrization=parametrization,
            cov=cov["self_attention"],
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)

        self.mlp = MLPBlock(
            hidden_dim,
            mlp_dim,
            dropout,
            cov=cov["mlp"],
            parametrization=parametrization,
        )

    def forward(
        self,
        input: Float[Tensor, "*sample batch_size seq_length hidden_dim"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        x = self.ln_1(input)
        x = self.self_attention(
            x,
            x,
            x,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(
            y,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        return x + y


class Encoder(bnn.BNNMixin, nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: (
            params.FactorizedCovariance
            | dict[params.FactorizedCovariance]
            | dict[dict[params.FactorizedCovariance]]
            | None
        ) = None,
    ):
        super().__init__()
        cov = check_cov(cov, [f"layers.encoder_layer_{i}" for i in range(num_layers)])

        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                parametrization=parametrization,
                cov=cov[f"layers.encoder_layer_{i}"],
            )
        self.layers = bnn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def reset_parameters(self) -> None:
        """Reset the parameters of the module and set the parametrization of all children
        to the parametrization of the module.

        Needs to be implemented because Encoder has direct parameters.
        """

        # direct parameters
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)  # from BERT

        # child modules
        self.layers.parametrization = self.parametrization
        self.layers.reset_parameters()

        reset_parameters_of_torch_module(self.ln, parametrization=self.parametrization)

    def parameters_and_lrs(
        self,
        lr: float,
        optimizer: Literal["SGD", "Adam"],
    ) -> list[dict[str, Tensor | float]]:
        """Get the parameters of the module and their learning rates for the chosen optimizer
        and the parametrization of the module.

        Needs to be implemented because Encoder has direct parameters.

        :param lr: The global learning rate.
        :param optimizer: The optimizer being used.
        """

        param_groups = []

        # direct parameters
        param_groups += [
            {
                "name": "pos_embedding",
                "params": [self.pos_embedding],
                "lr": lr,
            }
        ]

        # child modules
        param_groups += self.layers.parameters_and_lrs(lr=lr, optimizer=optimizer)
        param_groups += parameters_and_lrs_of_torch_module(
            self.ln,
            lr=lr,
            parametrization=self.parametrization,
            optimizer=optimizer,
        )

        return param_groups

    def forward(
        self,
        input: Float[Tensor, "*sample batch_size seq_length hidden_dim"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        num_sample_dims = 0 if sample_shape is None else len(sample_shape)

        if sample_shape is not None:
            input = input + self.pos_embedding.expand(
                *sample_shape, *self.pos_embedding.shape
            )
        else:
            input = input + self.pos_embedding

        output = self.dropout(input)
        output = self.layers(
            output,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        output = bnn.batched_forward(self.ln, num_batch_dims=num_sample_dims + 1)(
            output
        )
        return output


class VisionTransformer(bnn.BNNMixin, nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929.

    :param in_size: Size of the input (i.e. image size).
    :param patch_size: Size of the patch
    :param num_layers: Number of layers in the encoder
    :param num_heads: Number of heads
    :param hidden_dim: Hidden size in encoder
    :param mlp_dim: Dimension of MLP block
    :param dropout: dropout probability, defaults to 0.0
    :param attention_dropout: attention dropout probability, defaults to 0.0
    :param out_size: Size of the output (i.e. number of classes).
    :param representation_size: size of pre-logits layer before output head
    :param norm_layer:  Normalization layer to use.
    :param conv_stem_configs: currently not supported in inferno ViT
    :param parametrization: he parametrization to use. Defines the initialization
        and learning rate scaling for the parameters of the module.
    :param cov: Covariance structure of the probabilistic layers.
    """

    def __init__(
        self,
        in_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        out_size: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[list[ConvStemConfig]] = None,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: (
            params.FactorizedCovariance
            | dict[params.FactorizedCovariance]
            | dict[dict[params.FactorizedCovariance]]
            | None
        ) = None,
    ):
        super().__init__(parametrization=parametrization)
        _log_api_usage_once(self)
        torch._assert(
            in_size % patch_size == 0, "Input shape indivisible by patch size!"
        )
        self.in_size = in_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.out_size = out_size
        self.representation_size = representation_size
        if norm_layer is nn.BatchNorm2d:
            raise ValueError(
                "BatchNorm is currently not supported due to incompatibility of "
                "torch.vmap with the 'running_stats' tracked by BatchNorm."
                "See also: https://pytorch.org/docs/stable/func.batch_norm.html#patching-batch-norm."
            )
        self.norm_layer = norm_layer
        cov = check_cov(cov, ["conv_proj", "encoder", "heads.pre_logits", "heads.head"])

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            raise NotImplementedError(
                "conv_stem_configs currently not supported in inferno "
                "because there is no Conv2dNormActivation implementation."
            )
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last",
                nn.Conv2d(
                    in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1
                ),
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = bnn.Conv2d(
                in_channels=3,
                out_channels=hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
                cov=cov["conv_proj"],
                parametrization=parametrization,
                layer_type="input",
            )

        seq_length = (in_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
            parametrization=parametrization,
            cov=cov["encoder"],
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = bnn.Linear(
                hidden_dim,
                out_size,
                parametrization=parametrization,
                cov=cov["heads.head"],
                layer_type="output",
            )
        else:
            heads_layers["pre_logits"] = bnn.Linear(
                hidden_dim,
                representation_size,
                parametrization=parametrization,
                cov=cov["heads.pre_logits"],
                layer_type="hidden",
            )
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = bnn.Linear(
                representation_size,
                out_size,
                parametrization=parametrization,
                cov=cov["heads.head"],
                layer_type="output",
            )

        self.heads = bnn.Sequential(heads_layers)

        # Reset parameters (note this replaces torchvision initialization)
        self.reset_parameters()

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        """Load a VisionTransformer model with pretrained weights.

        Depending on the ``in_size`` and ``out_size`` parameters, the first and last
        layers of the model are not initialized with the pretrained weights.

        :param in_size: Size of the input (i.e. image size).
        :param out_size: Size of the output (i.e. number of classes).
        :param weights: Pretrained weights to use.
        :param freeze: Whether to freeze the pretrained weights.
        """
        # Load and preprocess the pretrained weights
        pretrained_weights = weights.get_state_dict(progress=True)
        if in_size != 224:
            # Remove the first layer (conv_proj) from the pretrained weights
            del pretrained_weights["conv_proj.weight"]
            del pretrained_weights["conv_proj.bias"]

            # Remove the positional embeddings
            del pretrained_weights["encoder.pos_embedding"]

        if out_size != pretrained_weights["heads.head.weight"].shape[0]:
            # Remove the last layer (head) from the pretrained weights
            del pretrained_weights["heads.head.weight"]
            del pretrained_weights["heads.head.bias"]

        # Model
        model = cls(
            *args,
            **kwargs,
            in_size=in_size,
            out_size=out_size,
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            pretrained_weights, strict=False
        )

        if freeze:
            # Freeze the pretrained weights
            for name, param in model.named_parameters():
                if name.replace(".params", "") in pretrained_weights:
                    param.requires_grad = False

        return model

    def reset_parameters(self) -> None:
        """Reset the parameters of the module and set the parametrization of all children
        to the parametrization of the module.

        Needs to be implemented because VisionTransformer has direct parameters.
        """

        # direct parameters
        nn.init.zeros_(self.class_token)

        # child modules
        self.conv_proj.parametrization = self.parametrization
        self.encoder.parametrization = self.parametrization
        self.heads.parametrization = self.parametrization

        self.conv_proj.reset_parameters()
        self.encoder.reset_parameters()
        self.heads.reset_parameters()

    def parameters_and_lrs(
        self,
        lr: float,
        optimizer: Literal["SGD", "Adam"],
    ) -> list[dict[str, Tensor | float]]:
        """Get the parameters of the module and their learning rates for the chosen optimizer
        and the parametrization of the module.

        Needs to be implemented because VisionTransformer has direct parameters.

        :param lr: The global learning rate.
        :param optimizer: The optimizer being used.
        """

        param_groups = []

        # direct parameters
        param_groups += [
            {
                "name": "class_token",
                "params": [self.class_token],
                "lr": lr,
            }
        ]

        # child modules
        param_groups += self.conv_proj.parameters_and_lrs(lr=lr, optimizer=optimizer)
        param_groups += self.encoder.parameters_and_lrs(lr=lr, optimizer=optimizer)
        param_groups += self.heads.parameters_and_lrs(lr=lr, optimizer=optimizer)

        return param_groups

    def _process_input(
        self,
        x: torch.Tensor,
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> torch.Tensor:
        num_sample_dims = 0 if sample_shape is None else len(sample_shape)

        if input_contains_samples:
            n, c, h, w = x.shape[num_sample_dims:]
        else:
            n, c, h, w = x.shape

        p = self.patch_size
        torch._assert(
            h == self.in_size,
            f"Wrong image height! Expected {self.in_size} but got {h}!",
        )
        torch._assert(
            w == self.in_size,
            f"Wrong image width! Expected {self.in_size} but got {w}!",
        )
        n_h = h // p
        n_w = w // p

        # (*sample_shape, n, c, h, w) -> (*sample_shape, n, hidden_dim, n_h, n_w)
        x = self.conv_proj(
            x,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        if sample_shape is not None:
            x = x.reshape(*sample_shape, n, self.hidden_dim, n_h * n_w)
        else:
            x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (*sample_shape, n, hidden_dim, (n_h * n_w)) -> (*sample_shape, n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.transpose(-2, -1)

        return x

    def forward(
        self,
        x: Float[Tensor, "*sample batch *in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        num_sample_dims = 0 if sample_shape is None else len(sample_shape)

        # Reshape and permute the input tensor
        x = self._process_input(
            x,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        n = x.shape[num_sample_dims]

        # Expand the class token to the full batch
        if sample_shape is not None:
            batch_class_token = self.class_token.expand(*sample_shape, n, -1, -1)
        else:
            batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=-2)

        x = self.encoder(
            x,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )

        # Classifier "token" as used by standard language architectures
        x = torch.select(x, num_sample_dims + 1, 0)

        x = self.heads(
            x,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )

        return x


class ViT_B_16(VisionTransformer):
    """ViT_B_16

    :param **kwargs: Additional keyword arguments passed on to [``VisionTransformer``][inferno.models.VisionTransformer].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights = torchvision.models.ViT_B_16_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            weights=weights,
            freeze=freeze,
            *args,
            **kwargs,
        )


class ViT_B_32(VisionTransformer):
    """ViT_B_32

    :param **kwargs: Additional keyword arguments passed on to [``VisionTransformer``][inferno.models.VisionTransformer].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=32,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights = torchvision.models.ViT_B_32_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            weights=weights,
            freeze=freeze,
            *args,
            **kwargs,
        )


class ViT_L_16(VisionTransformer):
    """ViT_L_16

    :param **kwargs: Additional keyword arguments passed on to [``VisionTransformer``][inferno.models.VisionTransformer].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights = torchvision.models.ViT_L_16_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            weights=weights,
            freeze=freeze,
            *args,
            **kwargs,
        )


class ViT_L_32(VisionTransformer):
    """ViT_L_32

    :param **kwargs: Additional keyword arguments passed on to [``VisionTransformer``][inferno.models.VisionTransformer].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=32,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights = torchvision.models.ViT_L_32_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            weights=weights,
            freeze=freeze,
            *args,
            **kwargs,
        )


class ViT_H_14(VisionTransformer):
    """ViT_H_14

    :param **kwargs: Additional keyword arguments passed on to [``VisionTransformer``][inferno.models.VisionTransformer].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=14,
            num_layers=32,
            num_heads=16,
            hidden_dim=1280,
            mlp_dim=5120,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights = torchvision.models.ViT_H_14_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            weights=weights,
            freeze=freeze,
            *args,
            **kwargs,
        )


def check_cov(
    cov: params.FactorizedCovariance | dict[str, Any] | None,
    required_cov_keys: list[str],
):
    """
    Converts cov to a dictionary with required_cov_keys or fills in missing keys with default covariance None

    :param cov: covariance or dictionary of covariances or None
    :param required_cov_keys: covariance keys required by this module
    """
    if cov is None:
        cov = {key: None for key in required_cov_keys}
    elif isinstance(cov, params.FactorizedCovariance):
        cov = {key: copy.deepcopy(cov) for key in required_cov_keys}
    elif isinstance(cov, dict):

        # check for unexpected covs
        for key in cov.keys():
            if key not in required_cov_keys:
                raise ValueError(f"Covariance key {key} not recognized")

        # set missing covs to default value (None)
        for key in required_cov_keys:
            if key not in cov.keys():
                cov[key] = None
    return cov


def last_layer_cov(cov: params.FactorizedCovariance, num_layers: int):
    """Returns a dictionary of covariances that specifies a covariance
    in the conv_proj layer, the last encoder layer (mlp and attention), and last output layer.

    :param cov: covariance to use in all last layers
    :param num_layers: number of layers in the VisionTransformer
    """
    return {
        "conv_proj": copy.deepcopy(cov),
        "encoder": {f"layers.encoder_layer_{num_layers-1}": copy.deepcopy(cov)},
        "heads.head": copy.deepcopy(cov),
    }
