from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple, _single, _triple

from .. import params
from .bnn_mixin import BNNMixin

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class _ConvNd(BNNMixin, nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: str | tuple[int, ...],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        layer_type: Literal["input", "hidden", "output"] = "hidden",
        cov: params.FactorizedCovariance | None = None,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(parametrization=parametrization)

        # Check arguments
        if groups <= 0:
            raise ValueError("groups must be a positive integer.")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups.")

        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}."
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions."
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'."
            )

        # Attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )
        if layer_type not in ["input", "hidden", "output"]:
            raise ValueError(
                f"Invalid layer type '{layer_type}'. Must be one of ['input', 'hidden', 'output']."
            )
        self.layer_type = layer_type

        # Parameters
        weight_shape = (
            (in_channels, out_channels // groups, *kernel_size)
            if transposed
            else (out_channels, in_channels // groups, *kernel_size)
        )
        mean_param_dict = {
            **{"weight": torch.empty(weight_shape, **factory_kwargs)},
            **(
                {"bias": torch.empty(out_channels, **factory_kwargs)}
                if bias
                else {"bias": None}
            ),
        }
        if cov is None:
            self.params = nn.ParameterDict(mean_param_dict)
            self.params.cov = None
        else:
            self.params = params.GaussianParameter(
                mean=mean_param_dict,
                cov=cov,
            )

        self.reset_parameters()

    @property
    def weight(self) -> nn.Parameter:
        return self.params.weight

    @property
    def bias(self) -> nn.Parameter:
        return self.params.bias

    def reset_parameters(
        self,
    ) -> None:
        """Reset the parameters of the module."""
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.params.weight)

        # Mean parameters
        mean_parameter_scales = {}
        mean_parameter_scales["weight"] = self.parametrization.weight_init_scale(
            fan_in, fan_out, layer_type=self.layer_type
        )
        nn.init.normal_(self.params.weight, mean=0, std=mean_parameter_scales["weight"])

        if self.params.bias is not None:
            mean_parameter_scales["bias"] = self.parametrization.bias_init_scale(
                fan_in, fan_out, layer_type=self.layer_type
            )
            nn.init.normal_(self.params.bias, mean=0, std=mean_parameter_scales["bias"])

        # Covariance parameters
        if self.params.cov is not None:
            self.params.cov.reset_parameters(mean_parameter_scales)

    def parameters_and_lrs(
        self,
        lr: float,
        optimizer: Literal["SGD", "Adam"] = "SGD",
    ) -> list[dict[str, Tensor | float]]:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.params.weight)

        # Weights
        mean_parameter_lr_scales = {}
        mean_parameter_lr_scales["weight"] = self.parametrization.weight_lr_scale(
            fan_in, fan_out, optimizer=optimizer, layer_type=self.layer_type
        )
        param_groups = [
            {
                "name": "params.weight",
                "params": [self.params.weight],
                "lr": lr * mean_parameter_lr_scales["weight"],
            }
        ]

        # Bias
        if self.params.bias is not None:
            mean_parameter_lr_scales["bias"] = self.parametrization.bias_lr_scale(
                fan_in, fan_out, optimizer=optimizer, layer_type=self.layer_type
            )
            param_groups += [
                {
                    "name": "params.bias",
                    "params": [self.params.bias],
                    "lr": lr * mean_parameter_lr_scales["bias"],
                }
            ]

        # Covariance
        if self.params.cov is not None:
            for name, param in self.params.cov.named_parameters():
                lr_scaling = 1.0
                if "weight" in name:
                    lr_scaling = mean_parameter_lr_scales["weight"]
                elif "bias" in name:
                    lr_scaling = mean_parameter_lr_scales["bias"]
                param_groups += [
                    {
                        "name": "params.cov." + name,
                        "params": [param],
                        "lr": lr * lr_scaling * self.params.cov.lr_scaling[name],
                    }
                ]

        return param_groups

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):

        # Ensure compatibility with nn.Linear
        if prefix + "weight" in state_dict:
            state_dict[prefix + "params.weight"] = state_dict.pop(prefix + "weight")
        if prefix + "bias" in state_dict:
            state_dict[prefix + "params.bias"] = state_dict.pop(prefix + "bias")

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _conv_forward(
        self,
        input: Float[Tensor, "*batch in_channel *in_feature"],
        weight: Float[Tensor, "out_channel in_channels_per_group kernel_size"],
        bias: Float[Tensor, "out_channel"] | None,
    ) -> Float[Tensor, "*batch out_channel *out_feature"]:
        raise NotImplementedError

    def forward(
        self,
        input: Float[Tensor, "*sample batch *in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch out_channel *out_feature"]:

        if sample_shape is None:
            # Forward pass with mean parameters
            return self._conv_forward(input, self.weight, self.bias)

        if not input_contains_samples:
            # Repeat input for each desired sample
            input = input.expand(*sample_shape, *input.shape)

        if parameter_samples is None:
            if self.params.cov is None:
                # If there's no covariance, expand the mean parameters into parameter samples
                parameter_samples = {
                    "weight": self.params.weight.expand(
                        *sample_shape, *self.params.weight.shape
                    ),
                    "bias": (
                        self.params.bias.expand(*sample_shape, *self.params.bias.shape)
                        if self.bias is not None
                        else None
                    ),
                }
            else:
                # Sample one set of parameters per sample
                parameter_samples = self.params.sample(
                    sample_shape=sample_shape, generator=generator
                )

        # Forward pass with sampled parameters
        if torch.Size(sample_shape) == torch.Size([]):
            # For a single sample, call _conv_forward directly
            return self._conv_forward(
                input,
                parameter_samples["weight"],
                (parameter_samples["bias"] if self.bias is not None else None),
            )
        else:
            # In case we have one or more sample dimensions, flatten, vmap and unflatten into sample_shape
            flattened_conv_result = torch.vmap(
                self._conv_forward,
                in_dims=(0, 0, 0 if self.bias is not None else None),
            )(
                input.flatten(start_dim=0, end_dim=len(sample_shape) - 1),
                parameter_samples["weight"].flatten(
                    start_dim=0, end_dim=len(sample_shape) - 1
                ),
                (
                    parameter_samples["bias"].flatten(
                        start_dim=0, end_dim=len(sample_shape) - 1
                    )
                    if self.bias is not None
                    else None
                ),
            )

            return flattened_conv_result.unflatten(0, sample_shape)


class Conv1d(_ConvNd):
    r"""Applies a 1D convolution over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    :param in_channels:     Number of channels in the input image.
    :param out_channels:    Number of channels produced by the convolution.
    :param kernel_size:     Size of the convolving kernel.
    :param stride:          Stride of the convolution.
    :param padding:         Padding added to both sides of the input.
    :param dilation:        Spacing between kernel elements.
    :param groups:          Number of blocked connections from input channels to output channels.
    :param bias:            If ``True``, adds a learnable bias to the output.
    :param padding_mode:    ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default is ``'zeros'``.
    :param layer_type:      Type of the layer. Can be one of "input", "hidden", or "output".
        Controls the initialization and learning rate scaling of the parameters.
    :param cov:             The covariance of the parameters.
    :param parametrization: The parametrization to use. Defines the initialization
        and learning rate scaling for the parameters of the module.
    :param device:          The device on which to place the tensor.
    :param dtype:           The desired data type of the returned tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        layer_type: Literal["input", "hidden", "output"] = "hidden",
        cov: params.FactorizedCovariance | None = None,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            layer_type=layer_type,
            cov=cov,
            parametrization=parametrization,
            **factory_kwargs,
        )

    def _conv_forward(
        self,
        input: Float[Tensor, "*batch in_channel in_feature"],
        weight: Float[Tensor, "out_channel in_channels_per_group kernel_size"],
        bias: Float[Tensor, "out_channel"] | None,
    ) -> Float[Tensor, "*batch out_channel out_feature"]:
        if self.padding_mode != "zeros":
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
            padding = _single(0)
        else:
            padding = self.padding

        return F.conv1d(
            input, weight, bias, self.stride, padding, self.dilation, self.groups
        )


class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    :param in_channels:     Number of channels in the input image.
    :param out_channels:    Number of channels produced by the convolution.
    :param kernel_size:     Size of the convolving kernel.
    :param stride:          Stride of the convolution.
    :param padding:         Padding added to all four sides of the input.
    :param dilation:        Spacing between kernel elements.
    :param groups:          Number of blocked connections from input channels to output channels.
    :param bias:            If ``True``, adds a learnable bias to the output.
    :param padding_mode:    ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default is ``'zeros'``.
    :param layer_type:      Type of the layer. Can be one of "input", "hidden", or "output".
        Controls the initialization and learning rate scaling of the parameters.
    :param cov:             The covariance of the parameters.
    :param parametrization: The parametrization to use. Defines the initialization
        and learning rate scaling for the parameters of the module.
    :param device:          The device on which to place the tensor.
    :param dtype:           The desired data type of the returned tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        layer_type: Literal["input", "hidden", "output"] = "hidden",
        cov: params.FactorizedCovariance | None = None,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            layer_type=layer_type,
            cov=cov,
            parametrization=parametrization,
            **factory_kwargs,
        )

    def _conv_forward(
        self,
        input: Float[Tensor, "*batch in_channel in_feature0 in_feature1"],
        weight: Float[Tensor, "out_channel in_channels_per_group kernel_size"],
        bias: Float[Tensor, "out_channel"] | None,
    ) -> Float[Tensor, "*batch out_channel out_feature0 out_feature1"]:
        if self.padding_mode != "zeros":
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
            padding = _pair(0)
        else:
            padding = self.padding

        return F.conv2d(
            input, weight, bias, self.stride, padding, self.dilation, self.groups
        )


class Conv3d(_ConvNd):
    r"""Applies a 3D convolution over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)

    where :math:`\star` is the valid 3D `cross-correlation`_ operator

    :param in_channels:     Number of channels in the input image.
    :param out_channels:    Number of channels produced by the convolution.
    :param kernel_size:     Size of the convolving kernel.
    :param stride:          Stride of the convolution.
    :param padding:         Padding added to all six sides of the input.
    :param dilation:        Spacing between kernel elements.
    :param groups:          Number of blocked connections from input channels to output channels.
    :param bias:            If ``True``, adds a learnable bias to the output.
    :param padding_mode:    ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default is ``'zeros'``.
    :param layer_type:      Type of the layer. Can be one of "input", "hidden", or "output".
        Controls the initialization and learning rate scaling of the parameters.
    :param cov:             The covariance of the parameters.
    :param parametrization: The parametrization to use. Defines the initialization
        and learning rate scaling for the parameters of the module.
    :param device:          The device on which to place the tensor.
    :param dtype:           The desired data type of the returned tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: str | _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        layer_type: Literal["input", "hidden", "output"] = "hidden",
        cov: params.FactorizedCovariance | None = None,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            layer_type=layer_type,
            cov=cov,
            parametrization=parametrization,
            **factory_kwargs,
        )

    def _conv_forward(
        self,
        input: Float[Tensor, "*batch in_channel in_feature0 in_feature1"],
        weight: Float[Tensor, "out_channel in_channels_per_group kernel_size"],
        bias: Float[Tensor, "out_channel"] | None,
    ) -> Float[Tensor, "*batch out_channel out_feature0 out_feature1"]:
        if self.padding_mode != "zeros":
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
            padding = _triple(0)
        else:
            padding = self.padding

        return F.conv3d(
            input, weight, bias, self.stride, padding, self.dilation, self.groups
        )
