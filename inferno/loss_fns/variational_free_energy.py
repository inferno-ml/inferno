from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from inferno import bnn

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


def _kl_normal_normal(
    loc_p: Float[Tensor, "parameter"],
    scale_p: Float[Tensor, "parameter parameter"],
    loc_q: Float[Tensor, "parameter"],
    scale_q: Float[Tensor, "parameter parameter"],
    eps: float = 1e-6,
):
    """KL Divergence between two normal distributions.

    :param loc_p:      Location of the first distribution.
    :param scale_p:    Scale of the first distribution.
    :param loc_q:      Location of the second distribution.
    :param scale_q:    Scale of the second distribution.
    :param eps:        Small value to avoid division by zero.
    """
    # Ensure that the variance is bounded away from zero
    # to avoid numerical instability
    var_p = scale_p.pow(2) + eps
    var_q = scale_q.pow(2) + eps
    # KL divergence between two normal distributions
    var_ratio = var_p / var_q
    quad_form = (loc_p - loc_q).pow(2) / var_q
    return 0.5 * (var_ratio + quad_form - 1.0 - var_ratio.log())


class VariationalFreeEnergy(nn.Module):
    """Variational Free Energy Loss.

    Computes the variational free energy loss for variational inference with the Kullback-Leibler
    regularization term computed in weight space. This is also known as the negative evidence lower
    bound (ELBO).

    :param nll:                 Loss function defining the negative log-likelihood.
    :param model:               The probabilistic model.
    :param prior_loc:           Location(s) of the prior Gaussian distribution.
    :param prior_scale:         Scale(s) of the prior Gaussian distribution.
    :param kl_weight:           Weight for the KL divergence term. If `None`, chooses the
        weight inversely proportional to the number of mean parameters.
    :param reduction:           Specifies the reduction to apply to the output:
        ````'mean'`` | ``'sum'``. ``'mean'``: the weighted mean of the output is taken,
        ``'sum'``: the output will be summed.
    """

    def __init__(
        self,
        nll: nn.modules.loss._Loss,
        model: bnn.BNNMixin,
        prior_loc: Float[Tensor, "parameter"] | None = None,
        prior_scale: (
            Float[Tensor, "parameter"] | None
        ) = None,  # NOTE: Assumes mean-field prior for now.
        kl_weight: float | None = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.nll = nll
        self.model = model
        self.numel_mean_parameters = sum(
            param.numel()
            for name, param in self.model.named_parameters()
            if param.requires_grad and "params." in name and "cov." not in name
        )
        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        if kl_weight is None:
            kl_weight = 1 / self.numel_mean_parameters
        self.kl_weight = kl_weight
        if reduction == "sum":
            raise NotImplementedError()
        self.nll.reduction = reduction
        self.reduction = reduction

    def forward(
        self,
        input: Float[Tensor, "*sample batch in_feature"],
        target: Float[Tensor, "batch out_feature"],
    ) -> Float[Tensor, ""]:
        # Expected negative log-likelihood (NLL)
        expected_nll = self.nll(
            input,
            target.expand((*input.shape[:-2], len(target))).reshape(-1),
        )  # NOTE: uses whatever reduction is set in the constructor

        # KL divergence regularization term
        variational_dist_mean = torch.concat(
            [
                param.ravel()
                for name, param in self.model.named_parameters()
                if param.requires_grad and "params." in name and "cov." not in name
            ]
        )
        variational_dist_scale = torch.concat(
            [
                param.ravel()
                for name, param in self.model.named_parameters()
                if param.requires_grad and "params.cov." in name
            ]
        )
        kl_divergence = _kl_normal_normal(
            loc_p=variational_dist_mean,
            scale_p=variational_dist_scale,
            loc_q=(
                self.prior_loc.to(device=variational_dist_mean.device)
                if self.prior_loc is not None
                else torch.zeros_like(variational_dist_mean)
            ),
            scale_q=(
                self.prior_scale.to(device=variational_dist_scale.device)
                if self.prior_scale is not None
                else torch.ones_like(variational_dist_scale)
            ),
        ).sum()
        if self.reduction == "mean":
            pass
        else:
            raise ValueError(f"Reduction '{self.reduction}' is not supported.")

        # Variational free energy (negative ELBO)
        return expected_nll + self.kl_weight * kl_divergence
