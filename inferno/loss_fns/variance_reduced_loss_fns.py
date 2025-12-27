from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from inferno import bnn

from .wrapped_torch_loss_fns import MSELoss, _predictions_and_expanded_targets

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
                   &= \mathbb{E}_{w_{1:L-1}}\big[\mathbb{E}_{w_L \mid w_{1:L-1}}[(f_w(x_n) - y_n)^2]\big]\\
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
        if reduction not in ["none", "sum", "mean"]:
            raise ValueError(
                f"Unsupported reduction '{self.reduction}'. Use 'none', 'sum', or 'mean'."
            )

        self._mse_loss = MSELoss(reduction="none")

    def forward(
        self,
        input_representation: Float[Tensor, "*sample batch *feature"],
        output_layer: bnn.BNNMixin,
        target: Float[Tensor, "batch *out_feature"],
    ):
        """Runs the forward pass.

        :param input_representation: (Penultimate layer) representation of input tensor. This is the representation produced by a
            forward pass through all hidden layers, which will be fed as inputs to the output layer in a forward pass.
        :param output_layer: Output layer of the model.
        :param target: Target tensor.
        """
        if not isinstance(output_layer, (bnn.Linear)):
            raise NotImplementedError(
                "Currently only supports linear Gaussian output layers."
            )

        # Compute predictive distribution conditioned on a sampled representation
        # NOTE: the current implementation is somewhat ad-hoc and only works for bnn.Linear output layers.
        # In the future this should simply rely on a .predictive() method in each BNNMixin module:
        # predictive_conditioned_on_representation = output_layer.predictive(
        #     input_representation
        # )

        mean_term = self._mse_loss(
            output_layer(input_representation, sample_shape=None),
            target,
        )

        if output_layer.params.cov is not None:
            if output_layer.bias is not None:
                if list(output_layer.params.cov.factor.keys())[0] == "bias":
                    # Ensure representation is padded with ones correctly depending on
                    # which cov parameters correspond to the bias
                    input_representation = torch.cat(
                        (
                            torch.ones((*input_representation.shape[0:-1], 1)),
                            input_representation,
                        ),
                        dim=-1,
                    )
                else:
                    input_representation = torch.cat(
                        (
                            input_representation,
                            torch.ones((*input_representation.shape[0:-1], 1)),
                        ),
                        dim=-1,
                    )

            variance_term = (
                torch.einsum(
                    "...f,fr->...r",
                    input_representation,
                    output_layer.params.cov._stacked_parameters(),
                )
                .pow(2)
                .sum(-1)
            )

            if output_layer.out_features == 1:
                variance_term = variance_term.unsqueeze(-1)
            elif output_layer.out_features > 1:
                raise NotImplementedError("Currently only supports 1D outputs.")

            loss = mean_term + variance_term
        else:
            loss = mean_term

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BCEWithLogitsLossVR(nn.modules.loss._Loss):
    r"""Binary Cross Entropy Loss with reduced variance for models with stochastic parameters.

    The loss on a single datapoint is given by

    $$
        \begin{align*}
            \ell_n &= \mathbb{E}_{w}[-\log p(y_n \mid f_w(x_n))]\\
                   &= -\mathbb{E}_{w}[y_n \log \sigma(f_w(x_n)) + (1 - y_n) \log \sigma(-f_w(x_n))]\\
                   &\leq \mathbb{E}_{w_{1:L-1}}\big[y_n \log(1+\mathbb{E}_{w_L \mid w_{1:L-1}}[\exp(-f_w(x_n))])\\
                    &\qquad+ (1 - y_n) \log(1+\mathbb{E}_{w_L \mid w_{1:L-1}}[\exp(f_w(x_n))])\big]
        \end{align*}
    $$

    which for a linear Gaussian output layer equals

    $$
        \begin{align*}
            \ell_n &\leq \mathbb{E}_{w_{1:L-1}}\big[ y_n \big(\log(1 + \exp(\mathbb{E}_{w_L \mid w_{1:L-1}}[-f_w(x_n)] + \frac{1}{2}\operatorname{Var}_{w_L \mid w_{1:L-1}}[f_w(x_n)]) \big) \\
                   &\qquad+ (1-y_n) \big(\log(1 + \exp(\mathbb{E}_{w_L \mid w_{1:L-1}}[f_w(x_n)] + \frac{1}{2}\operatorname{Var}_{w_L \mid w_{1:L-1}}[f_w(x_n)])\big)\big].
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
        target: Float[Tensor, "batch *out_feature"],
    ):
        """Runs the forward pass.

        :param input_representation: (Penultimate layer) representation of input tensor. This is the representation produced by a
            forward pass through all hidden layers, which will be fed as inputs to the output layer in a forward pass.
        :param output_layer: Output layer of the model.
        :param target: Target tensor.
        """
        if not isinstance(output_layer, (bnn.Linear)):
            raise NotImplementedError(
                "Currently only supports linear Gaussian output layers."
            )

        # Logits of model with mean parameters
        mean_logits = output_layer(input_representation, sample_shape=None)
        mean_logits = torch.concatenate(
            (torch.zeros_like(mean_logits), mean_logits), dim=-1
        )

        if output_layer.params.cov is not None:

            # Get representation and covariance factor
            if output_layer.bias is not None:
                bias_cov_factor = output_layer.params.cov.factor["bias"].unsqueeze(-2)
                weight_cov_factor = output_layer.params.cov.factor["weight"]

                if list(output_layer.params.cov.factor.keys())[0] == "bias":
                    # Ensure representation is padded with ones correctly depending on
                    # which cov parameters correspond to the bias
                    input_representation = torch.cat(
                        (
                            torch.ones((*input_representation.shape[0:-1], 1)),
                            input_representation,
                        ),
                        dim=-1,
                    )

                    cov_factor = torch.concatenate(
                        (bias_cov_factor, weight_cov_factor), dim=-2
                    )

                else:
                    input_representation = torch.cat(
                        (
                            input_representation,
                            torch.ones((*input_representation.shape[0:-1], 1)),
                        ),
                        dim=-1,
                    )

                    cov_factor = torch.concatenate(
                        (weight_cov_factor, bias_cov_factor), dim=-2
                    )

            else:
                cov_factor = output_layer.params.cov.factor["weight"]

            # Compute variance term
            variance_term = (
                torch.einsum(
                    "...f,cfr->...cr",
                    input_representation,
                    cov_factor,
                )
                .pow(2)
                .sum(-1)
            )
            # Add variance only to non-zero logits
            variance_term = torch.concatenate(
                (torch.zeros_like(variance_term), variance_term), dim=-1
            )

            # Compute loss
            loss = target * torch.logsumexp(
                -mean_logits + 0.5 * variance_term, dim=-1
            ) + (1 - target) * torch.logsumexp(
                mean_logits + 0.5 * variance_term, dim=-1
            )
        else:
            # Compute loss
            loss = target * torch.logsumexp(-mean_logits, dim=-1) + (
                1 - target
            ) * torch.logsumexp(mean_logits, dim=-1)

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
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
        target: Float[Tensor, "batch *out_feature"],
    ):
        """Runs the forward pass.

        :param input_representation: (Penultimate layer) representation of input tensor. This is the representation produced by a
            forward pass through all hidden layers, which will be fed as inputs to the output layer in a forward pass.
        :param output_layer: Output layer of the model.
        :param target: Target tensor.
        """
        if not isinstance(output_layer, (bnn.Linear)):
            raise NotImplementedError(
                "Currently only supports linear Gaussian output layers."
            )

        # Logits of model with mean parameters
        mean_logits = output_layer(input_representation, sample_shape=None)
        _, expanded_target = _predictions_and_expanded_targets(mean_logits, target)
        mean_logit_target_class = (
            torch.gather(
                mean_logits.flatten(0, -2), dim=-1, index=expanded_target.unsqueeze(-1)
            )
            .squeeze(-1)
            .unflatten(dim=0, sizes=mean_logits.shape[0:-1])
        )

        if output_layer.params.cov is not None:

            # Get representation and covariance factor
            if output_layer.bias is not None:
                bias_cov_factor = output_layer.params.cov.factor["bias"].unsqueeze(-2)
                weight_cov_factor = output_layer.params.cov.factor["weight"]

                if list(output_layer.params.cov.factor.keys())[0] == "bias":
                    # Ensure representation is padded with ones correctly depending on
                    # which cov parameters correspond to the bias
                    input_representation = torch.cat(
                        (
                            torch.ones((*input_representation.shape[0:-1], 1)),
                            input_representation,
                        ),
                        dim=-1,
                    )

                    cov_factor = torch.concatenate(
                        (bias_cov_factor, weight_cov_factor), dim=-2
                    )

                else:
                    input_representation = torch.cat(
                        (
                            input_representation,
                            torch.ones((*input_representation.shape[0:-1], 1)),
                        ),
                        dim=-1,
                    )

                    cov_factor = torch.concatenate(
                        (weight_cov_factor, bias_cov_factor), dim=-2
                    )

            else:
                cov_factor = output_layer.params.cov.factor["weight"]

            # Compute variance term
            variance_term = (
                torch.einsum(
                    "...f,cfr->...cr",
                    input_representation,
                    cov_factor,
                )
                .pow(2)
                .sum(-1)
            )

            # Compute loss
            loss = -mean_logit_target_class + torch.logsumexp(
                mean_logits + 0.5 * variance_term, dim=-1
            )
        else:
            # Compute loss
            loss = -mean_logit_target_class + torch.logsumexp(mean_logits, dim=-1)

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
