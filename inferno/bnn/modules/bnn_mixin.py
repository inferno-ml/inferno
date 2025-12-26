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
        if len(list(self.parameters(recurse=False))) > 0 or any(
            isinstance(child, (nn.ParameterDict, nn.ParameterList))
            for child in self.children()
        ):
            raise NotImplementedError(
                f"BNNMixin modules with parameters assigned to them must override 'reset_parameters()' "
                "to define how parameters should be initialized (depending on the parametrization). "
                "Be sure to also reset the parameters of any child modules according to the parametrization."
            )

        for child in self.children():
            if isinstance(child, BNNMixin):
                # Set the parametrization of all children to the parametrization of the parent module.
                child.parametrization = self.parametrization
                # Initialize the parameters of the child module.
                child.reset_parameters()
            else:
                reset_parameters_of_torch_module(
                    child, parametrization=self.parametrization
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
        # Check whether this module has any parameters itself (not just its children).
        if len(list(self.parameters(recurse=False))) > 0 or any(
            isinstance(child, (nn.ParameterDict, nn.ParameterList))
            for child in self.children()
        ):
            raise NotImplementedError(
                f"BNNMixin modules with parameters assigned to them must override 'parameters_and_lrs()' "
                "to define which learning rate scaling should be used according to the parametrization."
            )

        param_groups = []

        # Cycle through all children of the module and get their parameters and learning rates
        for child in self.children():

            # For layers with leaf parameters, return them with adjusted learning rate based on
            # the parametrization.
            if isinstance(child, BNNMixin):
                param_groups += child.parameters_and_lrs(lr=lr, optimizer=optimizer)
            else:
                param_groups += parameters_and_lrs_of_torch_module(
                    child,
                    lr=lr,
                    parametrization=self.parametrization,
                    optimizer=optimizer,
                )

        return param_groups

    def forward(
        self,
        input: Float[Tensor, "*sample batch *in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        """Forward pass of the module.

        :param input: Input tensor.
        :param sample_shape: Shape of samples. If None, runs a forward pass with just
            the mean parameters.
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
    """Reset the parameters of a torch.nn.Module and its children according to a given parametrization.

    :param module: The torch.nn.Module to reset the parameters of.
    :param parametrization: The parametrization to use.
    """
    module_parameter_names = [
        param_name for param_name, _ in module.named_parameters(recurse=False)
    ]
    if len(module_parameter_names) == 0:
        pass
    elif isinstance(
        module,
        (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
    ):
        # No need to change parameter initialization of layer norm according to Appendix B.1 of http://arxiv.org/abs/2203.03466
        module.reset_parameters()
    else:
        raise NotImplementedError(
            f"Cannot reset parameters of module: {module.__class__.__name__} "
            f"according to the {parametrization.__class__.__name__} parametrization."
        )

    # Reset parameters of child modules
    for child in module.children():

        if isinstance(child, BNNMixin):
            # Set the parametrization of all children to the parametrization of the parent module.
            child.parametrization = parametrization
            # Initialize the parameters of the child module.
            child.reset_parameters()
        else:
            reset_parameters_of_torch_module(child, parametrization=parametrization)


def parameters_and_lrs_of_torch_module(
    module: nn.Module,
    /,
    lr: float,
    parametrization: Parametrization,
    optimizer: Literal["SGD", "Adam"],
) -> list[dict[str, Tensor | float]]:
    """Get the parameters and their learning rates of module and its children for the chosen parametrization and optimizer.

    :param module: The torch.nn.Module to get the parameters and learning rates of.
    :param lr: The global learning rate.
    :param parametrization: The parametrization to use.
    :param optimizer: The optimizer being used.
    """
    param_groups = []
    module_parameter_names = [
        param_name for param_name, _ in module.named_parameters(recurse=False)
    ]
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
    else:
        raise NotImplementedError(
            f"Cannot set learning rates of module: {module.__class__.__name__} "
            f"according to the {parametrization.__class__.__name__} parametrization."
        )

    # Cycle through all children of the module and get their parameters and learning rates
    for child in module.children():
        if isinstance(child, BNNMixin):
            param_groups += child.parameters_and_lrs(lr=lr, optimizer=optimizer)
        else:
            param_groups += parameters_and_lrs_of_torch_module(
                child,
                lr=lr,
                parametrization=parametrization,
                optimizer=optimizer,
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
