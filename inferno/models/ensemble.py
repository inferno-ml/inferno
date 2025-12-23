"""Model ensembles.

The implementation largely follows the
[PyTorch documentation on ensembling](https://pytorch.org/tutorials/intermediate/ensembling.html).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Iterable

import torch
from torch import func, nn

from ..bnn import params
from ..bnn.modules.bnn_mixin import BNNMixin

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class Ensemble(BNNMixin, nn.Module):
    """An ensemble of models.

    This class ensembles multiple models with the same architecture by averaging their predictions.

    :param members: List of models to ensemble.
    """

    def __init__(self, members: Iterable[nn.Module]):

        # Parametrization
        parametrization = None
        for module in members:
            if hasattr(module, "parametrization"):
                # bnn.BNNMixin members
                if parametrization is None:
                    parametrization = copy.deepcopy(module.parametrization)
                elif not isinstance(module.parametrization, parametrization.__class__):
                    raise ValueError(
                        "All models in the ensemble must have the same parametrization."
                    )
            else:
                # nn.Module members
                parametrization = params.Standard()

        super().__init__(parametrization=parametrization)

        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        self.base_module = [copy.deepcopy(next(iter(members))).to(device="meta")]
        # NOTE: Assigning a list is a hack to avoid registering the base module as a submodule,
        # which can cause errors when using .apply, since it is on the meta device.

        self.members = nn.ModuleList(members)

    def forward(
        self,
        input: Float[Tensor, "*batch in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch out_feature"]:

        def functional_base_module_call(params, buffers, x):
            kwargs = (
                {
                    "sample_shape": sample_shape,
                    "generator": generator,
                    "input_contains_samples": input_contains_samples,
                    "parameter_samples": parameter_samples,
                }
                if isinstance(self.base_module[0], BNNMixin)
                else {}
            )
            return func.functional_call(
                self.base_module[0], (params, buffers), (x,), kwargs=kwargs
            )

        # Combine the states of each module by stacking parameters and buffers together.
        # NOTE: These are new leaf tensors in the graph, so they are independent of the member parameters!
        # In particular this means you cannot train the ensemble by using forward.
        params, buffers = func.stack_module_state(self.members)

        # Predict with each member of the ensemble in vectorized form.
        ensemble_forward_eval = torch.vmap(
            functional_base_module_call, in_dims=(0, 0, None), randomness="different"
        )(params, buffers, input)

        return ensemble_forward_eval
