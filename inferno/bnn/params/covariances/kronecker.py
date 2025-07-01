from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from .factorized import FactorizedCovariance

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class KroneckerCovariance(FactorizedCovariance):
    r"""Covariance of a Gaussian parameter with Kronecker structure.

    Assumes the covariance is given by a Kronecker product of two matrices of size
    equal to the number of inputs and outputs to the layer. Each Kronecker factor is
    assumed to be of rank $R \leq D$ where $D$ is either the input or
    output dimension of the layer.

    More precisely, the covariance is given by

    $$
    \begin{align*}
        \mathbf{\Sigma} &= \mathbf{C}_{\text{in}} \otimes \mathbf{C}_{\text{out}}\\
                        &= \mathbf{S}_{\text{in}}\mathbf{S}_{\text{in}}^\top \otimes \mathbf{S}_{\text{out}}\mathbf{S}_{\text{out}}^\top\\
                        &= (\mathbf{S}_{\text{in}} \otimes \mathbf{S}_{\text{out}}) (\mathbf{S}_{\text{in}}^\top \otimes \mathbf{S}_{\text{out}}^\top)
    \end{align*}
    $$

    where $\mathbf{S}_{\text{in}}$ and $\mathbf{S}_{\text{out}}$ are the
    low-rank factors of the Kronecker factors $\mathbf{C}_{\text{in}}$ and 
    $\mathbf{C}_{\text{out}}$.

    :param input_rank: Rank of the input Kronecker factor. If None, assumes full rank.
    :param output_rank: Rank of the output Kronecker factor. If None, assumes full rank.
    """

    def __init__(
        self,
        input_rank: int | None = None,
        output_rank: int | None = None,
    ):

        if input_rank is not None and input_rank < 1:
            raise ValueError(f"Input rank must be at least 1, but is {input_rank} < 1.")
        if output_rank is not None and output_rank < 1:
            raise ValueError(
                f"Output rank must be at least 1, but is {output_rank} < 1."
            )
        self.input_rank = input_rank
        self.output_rank = output_rank

        # rank(A \otimes B) = rank(A) * rank(B) where A and B are the Kronecker factors
        super().__init__(
            rank=(
                None
                if (input_rank is None) or (output_rank is None)
                else input_rank * output_rank
            )
        )

    def initialize_parameters(
        self,
        mean_parameters: dict[str, torch.Tensor],
    ) -> None:

        # Sizes of the Kronecker factors
        out_dim = mean_parameters["weight"].shape[0]
        in_dim = math.prod(mean_parameters["weight"].shape[1:])
        if "bias" in mean_parameters.keys() and mean_parameters["bias"] is not None:
            in_dim += mean_parameters["bias"].ndim
        self._fan_in = nn.init._calculate_fan_in_and_fan_out(mean_parameters["weight"])[
            0
        ]

        # Set the Kronecker factor ranks if not set
        if self.input_rank is None:
            self.input_rank = in_dim
        if self.output_rank is None:
            self.output_rank = out_dim
        if self.input_rank > in_dim:
            self.input_rank = in_dim
        if self.output_rank > out_dim:
            self.output_rank = out_dim

        # rank(A \otimes B) = rank(A) * rank(B) where A and B are the Kronecker factors
        self.rank = self.input_rank * self.output_rank

        # Initialize factor parameters
        input_factor_dict = {}
        for name, param in mean_parameters.items():
            if name == "bias":
                if mean_parameters["bias"] is not None:
                    input_factor_dict[name] = torch.empty(
                        1, self.input_rank, dtype=param.dtype, device=param.device
                    )
                else:
                    input_factor_dict[name] = None
            elif name == "weight":
                # Shapes (especially for convolutions) are inspired by K-FAC
                # See https://fdangel.com/posts/kfac_explained.html
                input_factor_dict[name] = torch.empty(
                    *param.shape[1:],
                    self.input_rank,
                    dtype=param.dtype,
                    device=param.device,
                )
                self.output_factor = nn.Parameter(
                    torch.empty(
                        param.shape[0],
                        self.output_rank,
                        dtype=param.dtype,
                        device=param.device,
                    )
                )
            else:
                raise NotImplementedError(
                    "Kronecker covariance currently only supports bias and weight parameters."
                )

        self.input_factor = nn.ParameterDict(input_factor_dict)

    def reset_parameters(
        self,
        mean_parameter_scales: dict[str, float] | float = 1.0,
    ) -> None:
        if isinstance(mean_parameter_scales, float):
            mean_parameter_scales = {
                name: mean_parameter_scales for name in self.input_factor.keys()
            }

        for name, param in self.input_factor.items():
            if param is not None:
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=mean_parameter_scales[name] / math.sqrt(self.input_rank),
                    # * math.pow(self._fan_in, 1 / 4),
                    # Last factor cancels with scaling in output_factor in forward pass,
                    # but approximately distributes mean parameter scaling across in- and output factor
                )

        nn.init.normal_(
            self.output_factor,
            mean=0.0,
            std=1 / math.sqrt(self.output_rank),  # * math.pow(self._fan_in, -1 / 4),
            # Last factor cancels with scaling in input_factor in forward pass,
            # but approximately distributes mean parameter scaling across in- and output factor),
        )

    def factor_matmul(
        self,
        input: Float[Tensor, "*sample parameter"],
        /,
        additive_constant: Float[Tensor, "*sample parameter"] | None = None,
    ) -> dict[str, Float[Tensor, "*sample parameter"]]:
        sample_shape = input.shape[:-1]

        # Stack factor parameters (input x rank, output x rank)
        stacked_input_factor, stacked_output_factor = self._stacked_parameters()

        # Multiply Kronecker factors with input
        # (A \otimes B) colvec(X) = colvec(B X A')
        result = stacked_output_factor @ (
            input.view(*sample_shape, self.input_rank, self.output_rank).mT
            @ stacked_input_factor.mT
        )
        result = result.mT.reshape(*sample_shape, -1)

        if additive_constant is not None:
            result = result + additive_constant

        # Split result into parameter shapes
        result_dict = {}
        starting_idx_weight_result = 0
        if "bias" in self.input_factor.keys() and self.input_factor["bias"] is not None:
            result_dict["bias"] = result[..., : self.output_factor.shape[0]]
            starting_idx_weight_result = self.output_factor.shape[0]
        if "weight" in self.input_factor.keys():
            result_dict["weight"] = result[..., starting_idx_weight_result:].view(
                *sample_shape,
                self.output_factor.shape[0],
                *self.input_factor["weight"].shape[:-1],
            )
        return result_dict

    @property
    def sample_scale(self) -> float:
        return 1.0 / math.sqrt(self.rank)

    def _stacked_parameters(self) -> tuple[Float[Tensor, "parameter"]]:
        stacked_input_factor = torch.vstack(
            [
                factor_param.view(-1, self.input_rank)
                for factor_param in self.input_factor.values()
                if factor_param is not None
            ]
        )
        return stacked_input_factor, self.output_factor

    def to_dense(self) -> Float[Tensor, "parameter parameter"]:
        stacked_parameters_input_factor, stacked_parameters_output_factor = (
            self._stacked_parameters()
        )
        return torch.kron(
            stacked_parameters_input_factor @ stacked_parameters_input_factor.mT,
            stacked_parameters_output_factor @ stacked_parameters_output_factor.mT,
        )

    @property
    def lr_scaling(self) -> dict[str, float]:
        """Compute the learning rate scaling for the covariance parameters."""
        return {
            **{
                "input_factor." + name: 1 / self.input_rank
                for name in self.input_factor.keys()
            },
            **{
                "output_factor": 1 / self.output_rank,
            },
        }
