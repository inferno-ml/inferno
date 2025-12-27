"""Residual Neural Networks (ResNets).

This implementation largely follows
[``torchvision.models.resnet``](https://github.com/pytorch/vision/blob/95f10a4ec9e43b2c8072ae5a68edd5700f9b1e45/torchvision/models/resnet.py).
"""

from __future__ import annotations

from collections.abc import Sequence
import copy
from typing import TYPE_CHECKING, Callable, Literal

import torch
from torch import nn
import torchvision

from .. import bnn
from ..bnn import params

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class ResNet(bnn.BNNMixin, nn.Module):
    """A residual neural network for image classification.

    :param out_size: Size of the output (i.e. number of classes).
    :param block: Block type to use.
    :param num_blocks_per_layer: Number of blocks per layer.
    :param zero_init_residual:
    :param groups: Number of groups for the convolutional layers.
    :param width_per_group: Width per group for the convolutional layers.
    :param replace_stride_with_dilation: Whether to replace the 2x2 stride with a dilated
        convolution. Must be a tuple of length 3.
    :param norm_layer: Normalization layer to use.
    :param architecture: Type of ResNet architecture. Either "imagenet" or "cifar".
    :param parametrization: The parametrization to use. Defines the initialization
        and learning rate scaling for the parameters of the module.
    :param cov: Covariance structure of the probabilistic layers.
    """

    def __init__(
        self,
        out_size: int,
        block: type["BasicBlock"] | type["Bottleneck"],
        num_blocks_per_layer: Sequence[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Sequence[bool] = (False, False, False),
        norm_layer: Callable[..., nn.Module] = lambda c: nn.GroupNorm(
            num_groups=32, num_channels=c
        ),
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: params.FactorizedCovariance | None = None,
    ) -> None:

        super().__init__(parametrization=parametrization)

        # Attributes
        if norm_layer is nn.BatchNorm2d:
            raise ValueError(
                "BatchNorm is currently not supported due to incompatibility of "
                "torch.vmap with the 'running_stats' tracked by BatchNorm."
                "See also: https://pytorch.org/docs/stable/func.batch_norm.html#patching-batch-norm."
            )
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "'replace_stride_with_dilation' should be "
                f"a 3-element tuple or list, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        # Layers
        if architecture == "cifar":
            self.conv1 = bnn.Conv2d(
                3,
                self.inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                cov=copy.deepcopy(cov),
                parametrization=self.parametrization,
                layer_type="input",
            )
        elif architecture == "imagenet":
            self.conv1 = bnn.Conv2d(
                3,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                cov=copy.deepcopy(cov),
                parametrization=self.parametrization,
                layer_type="input",
            )
        else:
            raise ValueError(
                f"Unknown architecture '{architecture}'. Expected 'cifar' or 'imagenet'."
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if architecture == "imagenet":
            self.optional_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.optional_pool = None
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks_per_layer[0],
            parametrization=parametrization,
            cov=(
                copy.deepcopy(cov)
                if isinstance(cov, params.DiagonalCovariance)
                else None
            ),
            layer_type="hidden",
        )
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks_per_layer[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            parametrization=parametrization,
            cov=(
                copy.deepcopy(cov)
                if isinstance(cov, params.DiagonalCovariance)
                else None
            ),
            layer_type="hidden",
        )
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks_per_layer[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            parametrization=parametrization,
            cov=(
                copy.deepcopy(cov)
                if isinstance(cov, params.DiagonalCovariance)
                else None
            ),
            layer_type="hidden",
        )
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks_per_layer[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            parametrization=parametrization,
            cov=(
                copy.deepcopy(cov)
                if isinstance(cov, params.DiagonalCovariance)
                else None
            ),
            layer_type="hidden",
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = bnn.Linear(
            512 * block.expansion,
            out_size,
            parametrization=parametrization,
            cov=copy.deepcopy(cov),
            layer_type="output",
        )

        # Reset parameters
        self.reset_parameters()

        # Zero-initialize the last bottleneck in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

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
        """Load a ResNet model with pretrained weights.

        Depending on the ``out_size`` and ``architecture`` parameters, the first and last
        layers of the model are not initialized with the pretrained weights.

        :param out_size: Size of the output (i.e. number of classes).
        :param weights: Pretrained weights to use.
        :param freeze: Whether to freeze the pretrained weights.
        :param architecture: Type of ResNet architecture. Either "imagenet" or "cifar".
        """
        # Load and preprocess the pretrained weights
        pretrained_weights = weights.get_state_dict(progress=True)
        if architecture != "imagenet":
            # Remove the first layer (conv1) from the pretrained weights
            del pretrained_weights["conv1.weight"]

        if out_size != pretrained_weights["fc.weight"].shape[0]:
            # Remove the last layer (fc) from the pretrained weights
            del pretrained_weights["fc.weight"]
            del pretrained_weights["fc.bias"]

        # Model
        model = cls(*args, **kwargs, out_size=out_size, architecture=architecture)
        missing_keys, unexpected_keys = model.load_state_dict(
            pretrained_weights, strict=False
        )

        if freeze:
            # Freeze the pretrained weights
            for name, param in model.named_parameters():
                if name.replace(".params", "") in pretrained_weights:
                    # TODO: Should the GroupNorm weight and bias be frozen? ResNet pretrained weights use BatchNorm
                    param.requires_grad = False

        return model

    def _make_layer(
        self,
        block: type["BasicBlock"] | type["Bottleneck"],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: params.FactorizedCovariance | None = None,
        layer_type: Literal["input", "hidden", "output"] = "hidden",
    ) -> nn.Sequential:
        """Make a ResNet layer for a given block type."""
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = bnn.Sequential(
                bnn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                    cov=(
                        copy.deepcopy(cov)
                        if isinstance(cov, params.DiagonalCovariance)
                        else copy.deepcopy(cov)
                    ),
                    parametrization=parametrization,
                    layer_type=layer_type,
                ),
                norm_layer(planes * block.expansion),
                parametrization=parametrization,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
                parametrization=parametrization,
                cov=(
                    copy.deepcopy(cov)
                    if isinstance(cov, params.DiagonalCovariance)
                    else copy.deepcopy(cov)
                ),
                layer_type=layer_type,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    parametrization=parametrization,
                    cov=(
                        copy.deepcopy(cov)
                        if isinstance(cov, params.DiagonalCovariance)
                        else copy.deepcopy(cov)
                    ),
                    layer_type=layer_type,
                )
            )

        return bnn.Sequential(*layers, parametrization=parametrization)

    def _forward_impl(
        self,
        input: Float[Tensor, "*sample batch *in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        """Implementation of forward which is compatible with TorchScript.

        It serves as an internal, TorchScript-friendly version of the standard forward method,
        allowing for model tracing and compilation for deployment while preserving the flexibility
        of the standard forward method in eager execution.
        """

        num_sample_dims = 0 if sample_shape is None else len(sample_shape)

        out = self.conv1(
            input,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        out = bnn.batched_forward(self.bn1, num_batch_dims=num_sample_dims + 1)(out)
        out = self.relu(out)
        if self.optional_pool is not None:
            out = bnn.batched_forward(
                self.optional_pool, num_batch_dims=num_sample_dims + 1
            )(out)

        out = self.layer1(
            out,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )
        out = self.layer2(
            out,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )
        out = self.layer3(
            out,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )
        out = self.layer4(
            out,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )

        out = bnn.batched_forward(self.avgpool, num_batch_dims=num_sample_dims + 1)(out)
        out = torch.flatten(out, -3)

        final_layer_forward_kwargs = {}
        if isinstance(self.fc, bnn.Linear):
            # This supports swapping out the final layer for a nn.Linear layer
            # e.g. in the case of Laplace approximation.
            final_layer_forward_kwargs = {
                "sample_shape": sample_shape,
                "generator": generator,
                "input_contains_samples": True,
                "parameter_samples": parameter_samples,
            }
        out = self.fc(
            out,
            **final_layer_forward_kwargs,
        )

        return out

    def forward(
        self,
        input: Float[Tensor, "*sample batch *in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        return self._forward_impl(
            input,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )


class BasicBlock(bnn.BNNMixin, nn.Module):
    """Basic block of a ResNet."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: params.FactorizedCovariance | None = None,
        layer_type: Literal["input", "hidden", "output"] = "hidden",
    ) -> None:
        super().__init__()

        # Attributes
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports 'groups=1' and 'base_width=64'.")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock.")

        # Layers
        self.conv1 = bnn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
            dilation=1,
            bias=False,
            parametrization=parametrization,
            cov=(
                copy.deepcopy(cov)
                if isinstance(cov, params.DiagonalCovariance)
                else copy.deepcopy(cov)
            ),
            layer_type=layer_type,
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = bnn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            dilation=1,
            bias=False,
            parametrization=parametrization,
            cov=(
                copy.deepcopy(cov)
                if isinstance(cov, params.DiagonalCovariance)
                else copy.deepcopy(cov)
            ),
            layer_type=layer_type,
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        input: Float[Tensor, "*sample batch *in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        num_sample_dims = 0 if sample_shape is None else len(sample_shape)

        identity = input

        out = self.conv1(
            input,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        out = bnn.batched_forward(self.bn1, num_batch_dims=num_sample_dims + 1)(out)
        out = self.relu(out)

        out = self.conv2(
            out,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )
        out = bnn.batched_forward(self.bn2, num_batch_dims=num_sample_dims + 1)(out)

        if self.downsample is not None:
            identity = self.downsample(
                input,
                sample_shape=sample_shape,
                generator=generator,
                input_contains_samples=True,
                parameter_samples=parameter_samples,
            )

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(bnn.BNNMixin, nn.Module):
    """Bottleneck block of a ResNet.

    Compared to the original implementation this places the stride for downsampling at the
    3x3 convolution(self.conv2), rather than at the first 1x1 convolution(self.conv1).
    This variant is also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch."
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: params.FactorizedCovariance | None = None,
        layer_type: Literal["input", "hidden", "output"] = "hidden",
    ) -> None:
        super().__init__()

        # Attributes
        width = int(planes * (base_width / 64.0)) * groups

        # Layers
        self.conv1 = bnn.Conv2d(
            inplanes,
            width,
            stride=1,
            kernel_size=1,
            bias=False,
            parametrization=parametrization,
            cov=(
                copy.deepcopy(cov)
                if isinstance(cov, params.DiagonalCovariance)
                else copy.deepcopy(cov)
            ),
            layer_type=layer_type,
        )
        self.bn1 = norm_layer(width)
        self.conv2 = bnn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            groups=groups,
            dilation=dilation,
            padding=dilation,
            bias=False,
            parametrization=parametrization,
            cov=(
                copy.deepcopy(cov)
                if isinstance(cov, params.DiagonalCovariance)
                else copy.deepcopy(cov)
            ),
            layer_type=layer_type,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = bnn.Conv2d(
            width,
            planes * self.expansion,
            stride=1,
            kernel_size=1,
            bias=False,
            parametrization=parametrization,
            cov=(
                copy.deepcopy(cov)
                if isinstance(cov, params.DiagonalCovariance)
                else copy.deepcopy(cov)
            ),
            layer_type=layer_type,
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        input: Float[Tensor, "*sample batch *in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        num_sample_dims = 0 if sample_shape is None else len(sample_shape)

        identity = input

        out = self.conv1(
            input,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        out = bnn.batched_forward(self.bn1, num_batch_dims=num_sample_dims + 1)(out)
        out = self.relu(out)

        out = self.conv2(
            out,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )
        out = bnn.batched_forward(self.bn2, num_batch_dims=num_sample_dims + 1)(out)
        out = self.relu(out)

        out = self.conv3(
            out,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )
        out = bnn.batched_forward(self.bn3, num_batch_dims=num_sample_dims + 1)(out)

        if self.downsample is not None:
            identity = self.downsample(
                input,
                sample_shape=sample_shape,
                generator=generator,
                input_contains_samples=True,
                parameter_samples=parameter_samples,
            )

        out += identity
        out = self.relu(out)

        return out


class ResNet18(ResNet):
    """ResNet-18

    :param **kwargs: Additional keyword arguments passed on to [``ResNet``][inferno.models.ResNet].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args, block=BasicBlock, num_blocks_per_layer=[2, 2, 2, 2], **kwargs
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ResNet18_Weights.DEFAULT,
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


class ResNet34(ResNet):
    """ResNet-34

    :param **kwargs: Additional keyword arguments passed on to [``ResNet``][inferno.models.ResNet].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args, block=BasicBlock, num_blocks_per_layer=[3, 4, 6, 3], **kwargs
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ResNet34_Weights.DEFAULT,
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


class ResNet50(ResNet):
    """ResNet-50

    :param **kwargs: Additional keyword arguments passed on to [``ResNet``][inferno.models.ResNet].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args, block=Bottleneck, num_blocks_per_layer=[3, 4, 6, 3], **kwargs
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ResNet50_Weights.DEFAULT,
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


class ResNet101(ResNet):
    """ResNet-101

    :param **kwargs: Additional keyword arguments passed on to [``ResNet``][inferno.models.ResNet].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args, block=Bottleneck, num_blocks_per_layer=[3, 4, 23, 3], **kwargs
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ResNet101_Weights.DEFAULT,
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


class ResNeXt50_32X4D(ResNet):
    """ResNext-50 (32x4d)

    :param **kwargs: Additional keyword arguments passed on to [``ResNet``][inferno.models.ResNet].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            block=Bottleneck,
            num_blocks_per_layer=[3, 4, 6, 3],
            groups=32,
            width_per_group=4,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT,
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


class ResNeXt101_32X8D(ResNet):
    """ResNext-101 (32x8d)

    :param **kwargs: Additional keyword arguments passed on to [``ResNet``][inferno.models.ResNet].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            block=Bottleneck,
            num_blocks_per_layer=[3, 4, 23, 3],
            groups=32,
            width_per_group=8,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ResNeXt101_32X8D_Weights.DEFAULT,
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


class ResNeXt101_64X4D(ResNet):
    """ResNext-101 (32x4d)

    :param **kwargs: Additional keyword arguments passed on to [``ResNet``][inferno.models.ResNet].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            block=Bottleneck,
            num_blocks_per_layer=[3, 4, 23, 3],
            groups=64,
            width_per_group=4,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT,
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


class WideResNet50(ResNet):
    """WideResNet-50-2

    Architecture described in [Wide Residual Networks](https://arxiv.org/abs/1605.06431). The model is the same
    as a ResNet except for the bottleneck number of channels which is twice larger in every block.

    :param **kwargs: Additional keyword arguments passed on to [``ResNet``][inferno.models.ResNet].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            block=Bottleneck,
            num_blocks_per_layer=[3, 4, 6, 3],
            width_per_group=64 * 2,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.Wide_ResNet50_2_Weights.DEFAULT,
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


class WideResNet101(ResNet):
    """WideResNet-101-2

    Architecture described in [Wide Residual Networks](https://arxiv.org/abs/1605.06431). The model is the same
    as a ResNet except for the bottleneck number of channels which is twice larger in every block.

    :param **kwargs: Additional keyword arguments passed on to [``ResNet``][inferno.models.ResNet].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            block=Bottleneck,
            num_blocks_per_layer=[3, 4, 23, 3],
            width_per_group=64 * 2,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        out_size: int,
        architecture: Literal["imagenet", "cifar"] = "imagenet",
        weights: torchvision.models.Weights = torchvision.models.Wide_ResNet101_2_Weights.DEFAULT,
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
