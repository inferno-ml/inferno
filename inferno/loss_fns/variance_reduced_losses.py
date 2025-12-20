from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import torch
from torch import nn

from inferno import bnn

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class VRMSELoss(nn.modules.loss._Loss):
    r"""Mean-Squared Error Loss with reduced variance for models with stochastic parameters.

    The loss on a single datapoint for models is given by

    $$
        \begin{align*}
            \ell_n &= \mathbb{E}_{w}[(f_w(x_n) - y_n)^2]\\
                   &= \mathbb{E}_{w_{1:L-1}}\big[(\mathbb{E}_{w_L \mid w_{1:L-1}}[f_w(x_n)] - y_n)^2 + \operatorname{Var}_{w_L \mid w_{1:L-1}}[f_w(x_n)]\big]\\
        \end{align*}
    $$

    which results in variance reduction for models with stochastic parameters compared to using [``inferno.loss_fns.MSELoss``][] which directly
    computes a Monte-Carlo approximation of the first line above.

    The ``reduction`` is applied over all sample and batch dimensions.

    :param reduction: Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied,
        ``'mean'``: the weighted mean of the output is taken,
        ``'sum'``: the output will be summed.
    """

    def __init__(
        self,
        reduction: Literal["none", "sum", "mean"] = "mean",
    ):
        super().__init__(reduction=reduction)

    def forward(
        self,
        representation: Callable[
            [Float[Tensor, "*sample batch *in_feature"]],
            Float[Tensor, "*sample batch *feature"],
        ],
        linear_output_layer_predictive: Callable[
            [Float[Tensor, "*sample batch *feature"]],
            Float[
                Tensor, "*sample batch *out_feature"
            ],  # TODO: should return distribution instead!
        ],
        input: Float[Tensor, "*sample batch *in_feature"],
        target: Float[Tensor, "*sample batch *out_feature"],
    ):
        raise NotImplementedError

    # TODO: Alternative, just let MSELoss check whether BNNMixin model has .representation function and use info from that?
    # TODO: Give (each?) BNNMixin a representation function and a function which returns mean and (co-)variance of the output layer conditioned on a
    # representation.
    # or just a conditional_predictive(x, sample_shape=(), diag_variance=True,...) which automatically conditions on a sample of the representation?
    # We also want a mode that just returns the prediction of using only mean parameters. This can be used if the last layer is linear.
    # Maybe do this if sample_shape = None? Or introduce mean_param_only = False?

    def forward_a(self, model: bnn.BNNMixin, input: Tensor, target: Tensor) -> Tensor:

        # Get samples of representation
        # Give each BNNMixin a representation function which needs to be overridden?
        # Standard layers just pass through the representation
        # Sequential just

        # Get last layer mean and cov of parameters

        raise NotImplementedError
