from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .parameter import BNNParameter

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from .covariances.factorized import FactorizedCovariance


class GaussianParameter(BNNParameter):
    """Parameter of a BNN module with Gaussian distribution.

    :param mean: Mean of the Gaussian distribution.
    :param cov: Covariance of the Gaussian distribution.
    """

    def __init__(
        self,
        mean: Float[Tensor, "parameter"] | dict[str, Float[Tensor, "parameter"]],
        cov: FactorizedCovariance,
    ):
        if not isinstance(mean, dict):
            mean = {"mean": mean}

        super().__init__(hyperparameters=mean)

        self.cov = cov
        self.cov.initialize_parameters(mean)

    def sample(
        self,
        sample_shape: torch.Size = torch.Size([]),
        generator: torch.Generator | None = None,
    ) -> (
        Float[Tensor, "*sample parameter"]
        | dict[str, Float[Tensor, "*sample parameter"]]
    ):
        # Sample from standard normal distribution
        standard_normal_sample = torch.randn(
            sample_shape + (self.cov.rank,),
            dtype=next(self.cov.parameters()).dtype,
            device=next(self.cov.parameters()).device,
            generator=generator,
        )

        # Transform the standard normal sample to the correct mean and covariance
        mean_params = {
            name: tens
            for name, tens in self.named_parameters()
            if "cov." not in name and "temperature" not in name
        }

        mean_params_stacked = torch.hstack(
            [tens.view(-1) for tens in mean_params.values()]
        )

        # Scale with inverse temperature if not training and the parameters are in the output layer
        if hasattr(self, "temperature") and not self.training:
            mean_params_stacked = mean_params_stacked / self.temperature

        return self.cov.factor_matmul(
            standard_normal_sample,
            additive_constant=mean_params_stacked,
        )
