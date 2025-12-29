from collections import OrderedDict
import copy
from functools import partial
import math
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple, Optional

from jaxtyping import Float
import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.ops.misc import MLP, Conv2dNormActivation
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once

from .. import bnn
from ..bnn import params
from ..models import MLP

if TYPE_CHECKING:
    pass


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

        for m in self.modules():
            if isinstance(m, bnn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

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

        if cov is None:
            cov = {key: None for key in ["self_attention", "mlp"]}
        elif isinstance(cov, params.FactorizedCovariance):
            cov = {key: copy.deepcopy(cov) for key in ["self_attention", "mlp"]}

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
        sample_shape: torch.Size = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ):
        # torch._assert(
        #    input.dim() == 3,
        #    f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        # )
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

        if cov is None:
            cov = {
                key: None for key in [f"encoder_layer_{i}" for i in range(num_layers)]
            }
        elif isinstance(cov, params.FactorizedCovariance):
            cov = {
                key: copy.deepcopy(cov)
                for key in [f"encoder_layer_{i}" for i in range(num_layers)]
            }

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
                cov=cov[f"encoder_layer_{i}"],
            )
        self.layers = bnn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(
        self,
        input: Float[Tensor, "*sample batch_size seq_length hidden_dim"],
        /,
        sample_shape: torch.Size = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ):
        # torch._assert(
        #    input.dim() == 3,
        #    f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        # )
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
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

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
        self.norm_layer = norm_layer

        if cov is None:
            cov = {key: None for key in ["conv_proj", "encoder", "pre_logits", "head"]}
        elif isinstance(cov, params.FactorizedCovariance):
            cov = {
                key: copy.deepcopy(cov)
                for key in ["conv_proj", "encoder", "pre_logits", "head"]
            }

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
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
                cov=cov["head"],
                layer_type="output",
            )
        else:
            heads_layers["pre_logits"] = bnn.Linear(
                hidden_dim,
                representation_size,
                parametrization=parametrization,
                cov=cov["pre_logits"],
                layer_type="hidden",
            )
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = bnn.Linear(
                representation_size,
                out_size,
                parametrization=parametrization,
                cov=cov["head"],
                layer_type="output",
            )

        self.heads = bnn.Sequential(heads_layers)

        if isinstance(self.conv_proj, bnn.Conv2d):
            # Init the patchify stem
            fan_in = (
                self.conv_proj.in_channels
                * self.conv_proj.kernel_size[0]
                * self.conv_proj.kernel_size[1]
            )
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(
            self.conv_proj.conv_last, nn.Conv2d
        ):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight,
                mean=0.0,
                std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels),
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(
            self.heads.pre_logits, bnn.Linear
        ):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(
                self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in)
            )
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, bnn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    @classmethod
    def from_pretrained_weights(
        cls,
        out_size: int,
        weights: torchvision.models.Weights,
        freeze: bool = False,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        *args,
        **kwargs,
    ):
        """Load a VisionTransformer model with pretrained weights.

        Depending on the ``out_size`` and ``architecture`` parameters, the first and last
        layers of the model are not initialized with the pretrained weights.

        :param out_size: Size of the output (i.e. number of classes).
        :param weights: Pretrained weights to use.
        :param freeze: Whether to freeze the pretrained weights.
        :param architecture: Type of VisionTransformer architecture. Either "imagenet" or "cifar".
        """
        # Load and preprocess the pretrained weights
        pretrained_weights = weights.get_state_dict(progress=True)
        if architecture != "imagenet":
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
            in_size=32 if architecture == "cifar" else 224,
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
        # x = x.permute(0, 2, 1)
        x = x.transpose(-2, -1)

        return x

    def forward(
        self,
        x: torch.Tensor,
        /,
        sample_shape: torch.Size = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ):
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
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ViT_B_16_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            out_size=out_size,
            architecture=architecture,
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
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ViT_B_32_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            out_size=out_size,
            architecture=architecture,
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
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ViT_L_16_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            out_size=out_size,
            architecture=architecture,
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
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ViT_L_32_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            out_size=out_size,
            architecture=architecture,
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
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ViT_H_14_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            out_size=out_size,
            architecture=architecture,
            weights=weights,
            freeze=freeze,
            *args,
            **kwargs,
        )
