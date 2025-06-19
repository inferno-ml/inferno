from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from .factorized import FactorizedCovariance

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class DiagonalCovariance(FactorizedCovariance):
    r"""Covariance of a Gaussian parameter with diagonal structure."""

    def __init__(self):
        super().__init__(rank=None)

    def initialize_parameters(
        self,
        mean_parameters: dict[str, torch.Tensor],
    ) -> None:
        if self.rank is None:
            # Total number of mean parameters
            self.rank = sum(
                [tens.numel() for tens in mean_parameters.values() if tens is not None]
            )

        # Initialize scale parameters
        self.scale = nn.ParameterDict(
            {
                name: (
                    torch.empty(
                        *param.shape,
                        dtype=param.dtype,
                        device=param.device,
                    )
                    if param is not None
                    else None
                )
                for name, param in mean_parameters.items()
            }
        )

    def reset_parameters(
        self,
        mean_parameter_scales: dict[str, float] | float = 1.0,
    ) -> None:
        if isinstance(mean_parameter_scales, float):
            mean_parameter_scales = {
                name: mean_parameter_scales for name in self.scale.keys()
            }

        for name, param in self.scale.items():
            if param is not None:
                nn.init.normal_(param, mean=0, std=mean_parameter_scales[name])

    def factor_matmul(
        self,
        input: Float[Tensor, "*sample parameter"],
        /,
        additive_constant: Float[Tensor, "*sample parameter"] | None = None,
    ) -> dict[str, Float[Tensor, "*sample parameter"]]:
        sample_shape = input.shape[:-1]

        # Stack scale parameters
        stacked_parameters = self._stacked_parameters()

        # Multiply scale parameters with input
        result = torch.einsum("...p,p->...p", input, stacked_parameters)
        if additive_constant is not None:
            result = result + additive_constant

        # Split result into parameter shapes
        split_result = torch.tensor_split(
            result,
            list(
                np.cumsum(
                    [tens.numel() for tens in self.scale.values() if tens is not None]
                )[:-1]
            ),
            dim=-1,
        )

        result_dict = {}
        i = 0
        for name, param in self.scale.items():
            if param is not None:
                result_dict[name] = split_result[i].view(*sample_shape, *param.shape)
                i += 1

        return result_dict

    def _stacked_parameters(self) -> Float[Tensor, "parameter"]:
        """Stack parameters into a single tensor."""
        stacked_parameters = torch.hstack(
            [
                scale_param.view(-1)
                for scale_param in self.scale.values()
                if scale_param is not None
            ]
        )
        return stacked_parameters

    def to_dense(self) -> Float[Tensor, "parameter parameter"]:
        """Convert the covariance matrix to a dense representation."""
        stacked_parameters = self._stacked_parameters()
        return torch.diag(stacked_parameters**2)

    @property
    def lr_scaling(self) -> dict[str, float]:
        return {"scale." + name: 1.0 for name in self.scale.keys()}
