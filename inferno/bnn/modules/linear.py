from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from .. import params
from .bnn_mixin import BNNMixin

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class Linear(BNNMixin, nn.Module):
    """
    Applies an affine transformation to the input.

    :param in_features: Size of each input sample.
    :param out_features: Size of each output sample.
    :param bias: If set to ``False``, the layer will not learn an additive bias.
    :param layer_type: Type of the layer. Can be one of "input", "hidden", or "output".
        Controls the initialization and learning rate scaling of the parameters.
    :param cov: Covariance object for the parameters.
    :param parametrization: The parametrization to use. Defines the initialization
        and learning rate scaling for the parameters of the module.
    :param device: Device on which to instantiate the parameters.
    :param dtype: Data type of the parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        layer_type: Literal["input", "hidden", "output"] = "hidden",
        cov: params.FactorizedCovariance | None = None,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(parametrization=parametrization)

        self.in_features = in_features
        self.out_features = out_features

        if layer_type not in ["input", "hidden", "output"]:
            raise ValueError(
                f"Invalid layer type '{layer_type}'. Must be one of ['input', 'hidden', 'output']."
            )
        self.layer_type = layer_type

        # Weight and bias parameters
        mean_param_dict = {
            **{"weight": torch.empty((out_features, in_features), **factory_kwargs)},
            **(
                {"bias": torch.empty(out_features, **factory_kwargs)}
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

        # Temperature parameter
        if layer_type == "output":
            self.params.temperature = nn.Parameter(
                torch.ones(1, **factory_kwargs), requires_grad=False
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

        # Temperature parameter
        if hasattr(self.params, "temperature"):
            nn.init.constant_(self.params.temperature, 1.0)
            self.params.temperature.requires_grad = False

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
                "layer_type": self.layer_type,
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
                    "layer_type": self.layer_type,
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
                        "layer_type": self.layer_type,
                    }
                ]

        return param_groups

    def forward(
        self,
        input: Float[Tensor, "*sample *batch in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch out_feature"]:

        if (
            parameter_samples is None and self.params.cov is None
        ) or sample_shape is None:
            output = nn.functional.linear(input, self.params.weight, self.params.bias)

            # Scale with inverse temperature if not training and the parameters are in the output layer
            if hasattr(self.params, "temperature") and not self.training:
                output = output / self.params.temperature

            if not input_contains_samples and sample_shape is not None:
                # Repeat output for each desired sample
                output = output.expand(*sample_shape, *output.shape)

        else:
            if not input_contains_samples:
                # Repeat input for each desired sample
                input = input.expand(*sample_shape, *input.shape)

            batch_shape = input.shape[len(sample_shape) : -1]

            if parameter_samples is None:
                # Sample one set of parameters per sample
                parameter_samples = self.params.sample(
                    sample_shape=sample_shape, generator=generator
                )

            # Affine transformation of input via sampled weights and biases
            weight_samples = parameter_samples["weight"]
            sample_dim_idcs = list(range(len(sample_shape)))
            output = torch.einsum(
                weight_samples,
                [*sample_dim_idcs, len(sample_dim_idcs) + 1, len(sample_dim_idcs) + 2],
                input,
                [*sample_dim_idcs, ..., len(sample_dim_idcs) + 2],
                [
                    *sample_dim_idcs,
                    ...,
                    len(sample_dim_idcs) + 1,
                ],  # Ensure sample dimensions are in front, batch dimensions next and out_feature dimensions last
            )
            if "bias" in parameter_samples:
                bias_samples = parameter_samples["bias"]
                output = output + bias_samples.view(
                    *(
                        sample_shape
                        + len(batch_shape) * (1,)
                        + (bias_samples.shape[-1],)
                    )
                )

        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.params.bias is not None}"

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
        if (
            hasattr(self.params, "temperature")
            and prefix + "params.temperature" not in state_dict
        ):
            state_dict[prefix + "params.temperature"] = self.params.temperature.data

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
