from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class FactorizedCovariance(nn.Module):
    r"""Covariance of a Gaussian parameter with a factorized structure.

    Assumes the covariance is factorized as a product of a square matrix and its transpose.

    ..math::
        \mathbf{\Sigma} = \mathbf{S} \mathbf{S}^\top

    :param rank: Rank of the covariance matrix. If `None`, the rank is set to the total number
        of mean parameters.
    """

    def __init__(self, rank: int | None = None):
        super().__init__()
        if rank is not None and rank < 1:
            raise ValueError(f"Rank must be at least 1, but is {rank} < 1.")
        self.rank = rank

    def initialize_parameters(
        self,
        mean_parameters: dict[str, torch.Tensor],
    ) -> None:
        """Initialize the covariance parameters.

        :param mean_parameters: Mean parameters of the Gaussian distribution.
        :return: Covariance parameters.
        """
        # Total number of mean parameters
        numel_mean_parameters = sum(
            [tens.numel() for tens in mean_parameters.values() if tens is not None]
        )
        if self.rank is None or self.rank > numel_mean_parameters:
            self.rank = numel_mean_parameters

        # Initialize factor parameters
        self.factor = nn.ParameterDict(
            {
                name: (
                    torch.empty(
                        *param.shape,
                        self.rank,
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
        """Reset the parameters of the covariance matrix.

        Initalizes the parameters of the covariance matrix with a
        scale that is given by the mean parameter scales and a
        covariance-specific scaling that depends on the structure of the covariance matrix.

        :param mean_parameter_scales: Scales of the mean parameters. If a dictionary
            keys are the names of the mean parameters. If a float, all covariance
            parameters are initialized with the same scale.
        """
        if isinstance(mean_parameter_scales, float):
            mean_parameter_scales = {
                name: mean_parameter_scales for name in self.factor.keys()
            }

        for name, param in self.factor.items():
            if param is not None:
                nn.init.normal_(
                    param,
                    mean=0,
                    std=mean_parameter_scales[name] / math.sqrt(self.rank),
                )

    def factor_matmul(
        self,
        input: Float[Tensor, "*sample parameter"],
        /,
        additive_constant: Float[Tensor, "*sample parameter"] | None = None,
    ) -> dict[str, Float[Tensor, "*sample parameter"]]:
        """Multiply left factor of the covariance matrix with the input.

        :param input: Input tensor.
        :param additive_constant: Additive constant to be added to the output.
        """
        sample_shape = input.shape[:-1]

        # Stack factor parameters
        stacked_parameters = self._stacked_parameters()

        # Multiply factor parameters with input
        result = torch.einsum("...p,qp->...q", input, stacked_parameters)

        if additive_constant is not None:
            result = result + additive_constant

        # Split result into parameter shapes
        split_result = torch.tensor_split(
            result,
            list(
                np.cumsum(
                    [
                        tens[..., 0].numel()
                        for tens in self.factor.values()
                        if tens is not None
                    ]
                )[:-1]
            ),
            dim=-1,
        )

        result_dict = {}
        i = 0
        for name, param in self.factor.items():
            if param is not None:
                result_dict[name] = split_result[i].view(
                    *sample_shape, *param.shape[:-1]
                )
                i += 1

        return result_dict

    def _stacked_parameters(self) -> Float[Tensor, "parameter"]:
        """Stack parameters into a single tensor."""
        stacked_parameters = torch.vstack(
            [
                factor_param.view(-1, self.rank)
                for factor_param in self.factor.values()
                if factor_param is not None
            ]
        )
        return stacked_parameters

    def to_dense(self) -> Float[Tensor, "parameter parameter"]:
        """Convert the covariance matrix to a dense representation."""
        stacked_parameters = self._stacked_parameters()
        return stacked_parameters @ stacked_parameters.mT

    @property
    def lr_scaling(self) -> dict[str, float]:
        """Compute the learning rate scaling for the covariance parameters."""
        return {"factor." + name: 1 / self.rank for name in self.factor.keys()}
