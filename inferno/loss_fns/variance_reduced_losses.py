from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import distributions, nn

from inferno import bnn

from .wrapped_torch_loss_fns import _predictions_and_expanded_targets

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


__all__ = [
    "BCEWithLogitsLossVR",
    "CrossEntropyLossVR",
    "MSELossVR",
]


class MSELossVR(nn.modules.loss._Loss):
    r"""Mean-Squared Error Loss with reduced variance for models with stochastic parameters.

    The loss on a single datapoint is given by

    $$
        \begin{align*}
            \ell_n &= \mathbb{E}_{w}[(f_w(x_n) - y_n)^2]\\
                   &= \mathbb{E}_{w_{1:L-1}}\big[(\mathbb{E}_{w_L \mid w_{1:L-1}}[f_w(x_n)] - y_n)^2 
                        + \operatorname{Var}_{w_L \mid w_{1:L-1}}[f_w(x_n)]\big].
        \end{align*}
    $$

    For models with stochastic parameters, the conditional Monte-Carlo estimate results in variance reduction compared to using [``inferno.loss_fns.MSELoss``][] which directly computes a Monte-Carlo approximation of the expected loss.

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
        # representation: Callable[
        #     [
        #         Float[Tensor, "batch *in_feature"],
        #         torch.Size,
        #     ],
        #     Float[Tensor, "*sample batch *feature"],
        # ],
        # linear_output_layer_predictive: Callable[
        #     [Float[Tensor, "*sample batch *feature"]],
        #     Float[
        #         Tensor, "*sample batch *out_feature"
        #     ],  # TODO: should return distribution instead!
        # ],
        # output_layer_predictive_dist: torch.distributions.Distribution,
        # input: Float[Tensor, "*sample batch *in_feature"],
        input_representation: Float[Tensor, "*sample batch *feature"],
        output_layer: bnn.BNNMixin,
        target: Float[Tensor, "*sample batch *out_feature"],
    ):
        """Runs the forward pass.

        :param input_representation: (Penultimate layer) representation of input tensor. This is the representation produced by a
            forward pass through all hidden layers, which will be fed as inputs to the output layer in a forward pass.
        :param output_layer: Output layer of the model.
        :param target: Target tensor.
        """

        # Compute predictive distribution conditioned on a sampled representation
        predictive_conditioned_on_representation = output_layer.predictive(
            input_representation
        )

        # Compute loss
        loss = (
            nn.functional.mse_loss(
                *_predictions_and_expanded_targets(
                    predictive_conditioned_on_representation.mean, target
                )
            )
            + predictive_conditioned_on_representation.variance
        )  # TODO: replace with inferno loss function stored as an attribute

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        # TODO: ensure shape is correct when applying no reduction (sample, batch)

        return loss

    # TODO: Give models in inferno.models a .representation(input) or .representation(input) function
    # TODO: Give all bnn.BNNMixin layers a .predictive(input) -> torch.distributions.Distribution function and if Parameters are GaussianParameters
    # then ensure the predictive(input) function is implemented (conditional of samples of the input).
    # TODO: Implement forward pass with sample_shape=None as forward pass with just mean_params (and use that for efficiency in linear layers)
    # TODO: tests for .predictive, .representation, and all losses


class BCEWithLogitsLossVR(nn.modules.loss._Loss):
    r"""Binary Cross Entropy Loss with reduced variance for models with stochastic parameters.

    The loss on a single datapoint is given by

    $$
        \begin{align*}
            \ell_n &= \mathbb{E}_{w}[-\log p(y_n \mid f_w(x_n))]\\
                   &= -\mathbb{E}_{w}[y_n \log \sigma(f_w(x_n)) + (1 - y_n) \log (1 - \sigma(f_w(x_n))]\\
                   &\leq \mathbb{E}_{w_{1:L-1}}\bigg[ y_n \big(\mathbb{E}_{w_L \mid w_{1:L-1}}[-f_w(x_n)] + \log(1 + \mathbb{E}_{w_L \mid w_{1:L-1}}[\exp(-f_w(x_n))]\big) \\
                   &\qquad+ (1-y_n) \big(\mathbb{E}_{w_L \mid w_{1:L-1}}[f_w(x_n)] + \log(1 + \mathbb{E}_{w_L \mid w_{1:L-1}}[\exp(f_w(x_n))]\big)\bigg]
        \end{align*}
    $$

    which for a linear Gaussian output layer equals

    $$
        \begin{align*}
            \ell_n &\leq \mathbb{E}_{w_{1:L-1}}\bigg[ y_n \big(\mathbb{E}_{w_L \mid w_{1:L-1}}[-f_w(x_n)] + \log(1 + \exp(\mathbb{E}_{w_L \mid w_{1:L-1}}[-f_w(x_n)] + \frac{1}{2}\operatorname{Var}_{w_L \mid w_{1:L-1}}[f_w(x_n)]) \big) \\
                   &\qquad+ (1-y_n) \big(\mathbb{E}_{w_L \mid w_{1:L-1}}[f_w(x_n)] + \log(1 + \exp(\mathbb{E}_{w_L \mid w_{1:L-1}}[f_w(x_n)] + \frac{1}{2}\operatorname{Var}_{w_L \mid w_{1:L-1}}[f_w(x_n)])\big)\bigg].
        \end{align*}
    $$

    which defines an upper bound on the expected value of the cross entropy loss. 
    For models with stochastic parameters, this loss has lower variance in exchange for bias compared to [``inferno.loss_fns.CrossEntropyLoss``][], 
    which directly computes a Monte-Carlo approximation of the expected loss .
    
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
        input_representation: Float[Tensor, "*sample batch *feature"],
        output_layer: bnn.BNNMixin,
        target: Float[Tensor, "*sample batch *out_feature"],
    ):
        """Runs the forward pass.

        :param input_representation: (Penultimate layer) representation of input tensor. This is the representation produced by a
            forward pass through all hidden layers, which will be fed as inputs to the output layer in a forward pass.
        :param output_layer: Output layer of the model.
        :param target: Target tensor.
        """

        # Compute predictive distribution conditioned on a sampled representation
        predictive_conditioned_on_representation = output_layer.predictive(
            input_representation
        )

        # Compute loss
        if isinstance(predictive_conditioned_on_representation, distributions.Normal):
            loss = torch.gather(  # TODO: ensure target view is correct even when probs contains samples
                predictive_conditioned_on_representation.mean,
                dim=-1,
                index=target.view(-1, 1),
            ) + torch.logsumexp(
                predictive_conditioned_on_representation.mean
                + 0.5 * predictive_conditioned_on_representation.variance,
                dim=-1,
            )
        else:
            raise NotImplementedError

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        # TODO: ensure shape is correct when applying no reduction (sample, batch)

        return loss


class CrossEntropyLossVR(nn.modules.loss._Loss):
    r"""Cross Entropy Loss with reduced variance for models with stochastic parameters.

    The loss on a single datapoint is given by

    $$
        \begin{align*}
            \ell_n &= \mathbb{E}_{w}[-\log p(y_n \mid f_w(x_n))]\\
                   &= \mathbb{E}_{w}[-\log \operatorname{softmax}(f_w(x_n))_{y_n}]\\
                   &\leq \mathbb{E}_{w_{1:L-1}}\bigg[\mathbb{E}_{w_L \mid w_{1:L-1}}[-f_w(x_n)_{y_n}]
                        + \log \sum_{c=1}^C \mathbb{E}_{w_L \mid w_{1:L-1}}[\exp(f_w(x_n)_{c})]\bigg],
        \end{align*}
    $$

    which for a linear Gaussian output layer equals

    $$
        \begin{equation*}
            \ell_n \leq \mathbb{E}_{w_{1:L-1}}\bigg[\mathbb{E}_{w_L \mid w_{1:L-1}}[-f_w(x_n)_{y_n}]
                        + \operatorname{logsumexp}\big(\mathbb{E}_{w_L \mid w_{1:L-1}}[f_w(x_n)_{c}] 
                        + \frac{1}{2}\operatorname{Var}_{w_L \mid w_{1:L-1}}[f_w(x_n)_c]\big)\bigg].
        \end{equation*}
    $$

    This loss defines an upper bound on the expected value of the cross entropy loss and for models with 
    stochastic parameters has lower variance in exchange for bias compared to [``inferno.loss_fns.CrossEntropyLoss``][], 
    which directly computes a Monte-Carlo approximation of the expected loss.
    
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
        input_representation: Float[Tensor, "*sample batch *feature"],
        output_layer: bnn.BNNMixin,
        target: Float[Tensor, "*sample batch *out_feature"],
    ):
        """Runs the forward pass.

        :param input_representation: (Penultimate layer) representation of input tensor. This is the representation produced by a
            forward pass through all hidden layers, which will be fed as inputs to the output layer in a forward pass.
        :param output_layer: Output layer of the model.
        :param target: Target tensor.
        """

        # Compute predictive distribution conditioned on a sampled representation
        predictive_conditioned_on_representation = output_layer.predictive(
            input_representation
        )

        # Compute loss
        if isinstance(predictive_conditioned_on_representation, distributions.Normal):
            loss = torch.gather(  # TODO: ensure target view is correct even when probs contains samples
                predictive_conditioned_on_representation.mean,
                dim=-1,
                index=target.view(-1, 1),
            ) + torch.logsumexp(
                predictive_conditioned_on_representation.mean
                + 0.5 * predictive_conditioned_on_representation.variance,
                dim=-1,
            )
        else:
            raise NotImplementedError

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        # TODO: ensure shape is correct when applying no reduction (sample, batch)

        return loss
