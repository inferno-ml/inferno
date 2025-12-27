from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, OrderedDict, overload

import torch
from torch import nn

from ...params import BNNParameter, Parametrization
from ..bnn_mixin import BNNMixin, batched_forward

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class Sequential(BNNMixin, nn.Sequential):
    """A sequential container for modules.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.

    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``)

    :param *args: Any number of modules to add to the container.
    :param parametrization: The parametrization to use. If `None`, the
        parametrization of the modules in the container will be used. If
        a [``Parametrization``][inferno.bnn.params.Parametrization] object is passed,
        it will be used for all modules in the container.
    """

    @overload
    def __init__(
        self,
        *args: BNNMixin | nn.Module,
        parametrization: Parametrization | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        arg: OrderedDict[str, BNNMixin | nn.Module],
        parametrization: Parametrization | None = None,
    ) -> None: ...

    def __init__(self, *args, parametrization: Parametrization | None = None):
        super().__init__(parametrization=parametrization)

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

        if parametrization is not None:
            # Recursively set parametrization of children to the given parametrization
            self.parametrization = parametrization
            """Parametrization of the module."""

            # Reset parameters of all modules in the container
            self.reset_parameters()

    def forward(
        self,
        input: Float[Tensor, "*batch in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch out_feature"]:

        # Sequential forward passes through all modules in the container
        for module in self:

            if isinstance(module, BNNMixin):

                if any(
                    [
                        isinstance(submodule, BNNParameter)
                        for submodule in module.modules()
                    ]
                ):
                    # If the module contains BNN parameters, we need to sample

                    if not input_contains_samples and sample_shape is not None:

                        # Repeat input for each sample
                        input = input.expand(*sample_shape, *input.shape)
                        input_contains_samples = True

                    # Forward pass
                    input = module(
                        input,
                        sample_shape=sample_shape,
                        generator=generator,
                        input_contains_samples=input_contains_samples,
                        parameter_samples=parameter_samples,
                    )
                else:
                    # No BNN parameters, so we can call the module directly without expanding the input
                    input = module(
                        input,
                        sample_shape=(
                            sample_shape if input_contains_samples else torch.Size([])
                        ),
                        generator=generator,
                        input_contains_samples=input_contains_samples,
                        parameter_samples=parameter_samples,
                    )
            elif isinstance(module, nn.Module):
                if not input_contains_samples:
                    # No sample dimensions have been added yet, so we can call the module directly
                    input = module(input)
                else:
                    # Some torch.Modules allow for only a single batch dimension.
                    # Since we potentially operate on multiple batch dimensions (i.e *samples batch)
                    # we need to account for this to ensure interoperability with torch.Modules.
                    if isinstance(module, nn.Flatten):
                        new_start_dim = module.start_dim + (
                            len(sample_shape) if module.start_dim >= 0 else 0
                        )
                        new_end_dim = module.end_dim + (
                            len(sample_shape) if module.end_dim >= 0 else 0
                        )
                        input = input.flatten(
                            start_dim=new_start_dim, end_dim=new_end_dim
                        )
                    elif isinstance(module, nn.Unflatten):
                        new_dim = module.dim + (
                            len(sample_shape) if module.dim >= 0 else 0
                        )
                        input = input.unflatten(new_dim, module.unflattened_size)
                    elif isinstance(
                        module,
                        (nn.Linear, nn.Identity)
                        + tuple(
                            (  # Activation functions
                                module
                                for _, module in inspect.getmembers(
                                    torch.nn.modules.activation
                                )
                                if inspect.isclass(module)
                                and module.__name__
                                in torch.nn.modules.activation.__all__
                            )
                        ),
                    ):
                        # Modules which allow for arbitrary many batch dimensions
                        input = module(input)
                    else:
                        # Call the module's forward pass in batched mode
                        num_sample_dims = (
                            0 if sample_shape is None else len(sample_shape)
                        )
                        input = batched_forward(
                            module, num_batch_dims=num_sample_dims + 1
                        )(input)
            else:
                raise ValueError(
                    f"Sequential contains unsupported module type: {type(module)}"
                )

        if not input_contains_samples and sample_shape is not None:
            # In case there are no BNN layers, simply expand the output to the sample shape.
            input = input.expand(*sample_shape, *input.shape)

        return input
