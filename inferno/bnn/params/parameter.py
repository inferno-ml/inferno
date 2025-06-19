from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class BNNParameter(nn.ParameterDict, abc.ABC):

    def __init__(
        self, hyperparameters: dict[str, Float[Tensor, "*hyperparameter"]] | None = None
    ):
        super().__init__(hyperparameters)

    def sample(
        self,
        sample_shape: torch.Size = torch.Size([]),
        generator: torch.Generator | None = None,
    ) -> Float[Tensor, "*sample parameter"]:
        raise NotImplementedError
