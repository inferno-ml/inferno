from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Callable, Literal

import torch
from torch import nn

from ..params import MaximalUpdate, Parametrization

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class BNNMixin(abc.ABC):
    """Abstract mixin class turning a torch module into a Bayesian neural network module.

    :param parametrization: The parametrization to use. Defines the initialization
        and learning rate scaling for the parameters of the module.
    """

    def __init__(
        self, parametrization: Parametrization | None = MaximalUpdate(), *args, **kwargs
    ) -> None:
        self._parametrization = parametrization

        # Forward all unused arguments to constructors of other parent classes of child class.
        super().__init__(*args, **kwargs)

    @property
    def parametrization(self) -> Parametrization:
        """Parametrization of the module."""
        return self._parametrization

    @parametrization.setter
    def parametrization(self, new_parametrization) -> Parametrization:
        self._parametrization = new_parametrization

        # Set the parametrization of all children to the new parametrization.
        for layer in self.children():
            if isinstance(layer, BNNMixin):
                layer.parametrization = self.parametrization

    def reset_parameters(self) -> None:
        """Reset the parameters of the module and set the parametrization of all children
        to the parametrization of the module.

        This method should be implemented by subclasses to reset the parameters of the module.
        """
        for layer in self.children():
            if isinstance(layer, BNNMixin):
                # Set the parametrization of all children to the parametrization of the parent module.
                layer.parametrization = self.parametrization
                # Initialize the parameters of the child module.
                layer.reset_parameters()
            else:
                if hasattr(layer, "reset_parameters"):
                    reset_parameters_of_torch_module(
                        layer, parametrization=self.parametrization
                    )

    def parameters_and_lrs(
        self,
        lr: float,
        optimizer: Literal["SGD", "Adam"],
    ) -> list[dict[str, Tensor | float]]:
        """Get the parameters of the module and their learning rates for the chosen optimizer
        and the parametrization of the module.

        :param lr: The global learning rate.
        :param optimizer: The optimizer being used.
        """
        param_groups = []

        # Cycle through all children of the module and get their parameters and learning rates
        for layer in self.children():

            # For layers with leaf parameters, return them with adjusted learning rate based on
            # the parametrization.
            if isinstance(layer, BNNMixin):
                param_groups += layer.parameters_and_lrs(lr=lr, optimizer=optimizer)
            else:
                if len(list(layer.parameters())) > 0:
                    param_groups += parameters_and_lrs_of_torch_module(
                        layer,
                        lr=lr,
                        parametrization=self.parametrization,
                        optimizer=optimizer,
                    )

        return param_groups

    def forward(
        self,
        input: Float[Tensor, "*sample batch *in_feature"],
        /,
        sample_shape: torch.Size = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        """Forward pass of the module.

        :param input: Input tensor.
        :param sample_shape: Shape of samples.
        :param generator: Random number generator.
        :param input_contains_samples: Whether the input already contains
            samples. If True, the input is assumed to have ``len(sample_shape)``
            many leading dimensions containing input samples (typically
            outputs from previous layers).
        :param parameter_samples: Dictionary of parameter samples. Used to pass
            sampled parameters to the module. Useful to jointly sample parameters
            of multiple layers.
        """
        raise NotImplementedError


def reset_parameters_of_torch_module(
    module: nn.Module,
    /,
    parametrization: Parametrization,
) -> None:
    """Reset the parameters of a torch.nn.Module according to a given parametrization.

    :param module: The torch.nn.Module to reset the parameters of.
    :param parametrization: The parametrization to use.
    """
    module_parameter_names = [param_name for param_name, _ in module.named_parameters()]
    if len(module_parameter_names) == 0:
        return
    elif isinstance(
        module,
        (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
    ):
        # No need to change parameter initialization of layer norm according to Appendix B.1 of http://arxiv.org/abs/2203.03466
        module.reset_parameters()
    elif "weight" in module_parameter_names:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)

        nn.init.normal_(
            module.weight,
            mean=0,
            std=parametrization.weight_init_scale(
                fan_in=fan_in, fan_out=fan_out, layer_type="hidden"
            ),
        )
        if module.bias is not None:
            nn.init.normal_(
                module.bias,
                mean=0,
                std=parametrization.bias_init_scale(
                    fan_in=fan_in, fan_out=fan_out, layer_type="hidden"
                ),
            )
    else:
        raise NotImplementedError(
            f"Cannot reset parameters of module: {module.__class__.__name__} "
            f"according to the {parametrization.__class__.__name__} parametrization."
        )


def parameters_and_lrs_of_torch_module(
    module: nn.Module,
    /,
    lr: float,
    parametrization: Parametrization,
    optimizer: Literal["SGD", "Adam"],
) -> list[dict[str, Tensor | float]]:
    """Get the parameters and their learning rates for the chosen parametrization and optimizer.

    :param module: The torch.nn.Module to get the parameters and learning rates of.
    :param lr: The global learning rate.
    :param parametrization: The parametrization to use.
    :param optimizer: The optimizer being used.
    """
    param_groups = []
    module_parameter_names = [param_name for param_name, _ in module.named_parameters()]
    if len(module_parameter_names) == 0:
        pass
    elif isinstance(
        module,
        (
            nn.LayerNorm,
            nn.GroupNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
        ),
    ):
        fan_out = module.weight.shape.numel()
        param_groups + [
            {
                "params": [module.weight],
                "lr": lr
                * parametrization.weight_lr_scale(
                    fan_in=1.0,
                    fan_out=fan_out,
                    optimizer=optimizer,
                    layer_type="input",
                ),
            }
        ]

        if module.bias is not None:
            param_groups + [
                {
                    "params": [module.bias],
                    "lr": lr
                    * parametrization.bias_lr_scale(
                        fan_in=1.0,
                        fan_out=fan_out,
                        optimizer=optimizer,
                        layer_type="input",
                    ),
                }
            ]
    elif "weight" in module_parameter_names:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)

        param_groups + [
            {
                "params": [module.weight],
                "lr": lr
                * parametrization.weight_lr_scale(
                    fan_in=fan_in, fan_out=fan_out, optimizer=optimizer
                ),
            }
        ]

        if module.bias is not None:
            param_groups + [
                {
                    "params": [module.bias],
                    "lr": lr
                    * parametrization.bias_lr_scale(
                        fan_in=fan_in, fan_out=fan_out, optimizer=optimizer
                    ),
                }
            ]
    else:
        raise NotImplementedError(
            f"Cannot set learning rates of module: {module.__class__.__name__} "
            f"according to the {parametrization.__class__.__name__} parametrization."
        )

    return param_groups


def batched_forward(obj: nn.Module, num_batch_dims: int) -> Callable[
    [Float[Tensor, "*sample batch *in_feature"]],
    Float[Tensor, "*sample batch *out_feature"],
]:
    """Call a torch.nn.Module on inputs with arbitrary many batch dimensions rather than
    just a single one.

    This is useful to extend the functionality of a torch.nn.Module to work with arbitrary
    many batch dimensions, for example arbitrary many sampling dimensions.

    :param obj: The torch.nn.Module to call.
    :param num_batch_dims: The number of batch dimensions.
    """
    if num_batch_dims < 0:
        raise ValueError(
            f"num_batch_dims must be non-negative, but is {num_batch_dims}."
        )
    if num_batch_dims <= 1:
        return obj.__call__

    def batched_forward_helper(
        input: Float[Tensor, "*sample *batch *in_feature"],
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        flattened_input = input.flatten(start_dim=0, end_dim=num_batch_dims - 2)
        flattened_output = torch.vmap(obj.__call__, in_dims=0, out_dims=0)(
            flattened_input
        )
        return flattened_output.unflatten(0, input.shape[: num_batch_dims - 1])

    return batched_forward_helper
