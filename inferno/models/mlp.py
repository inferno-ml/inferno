from __future__ import annotations

import copy
from typing import Callable

from torch import nn

from inferno import bnn
from inferno.bnn import params


class MLP(bnn.Sequential):
    """A fully-connected feedforward neural network with the same activation function in each layer.

    :param in_size:             Size of the input.
    :param hidden_sizes:        List of hidden layer sizes.
    :param out_size:            Size of the output (e.g. number of classes).
    :param norm_layer:          Normalization layer which will be stacked on top of the linear layer.
    :param activation_layer:    Activation function following a linear layer.
    :param inplace:             Whether to apply the activation function and dropout inplace.
            Default is ``None``, which uses the respective default values.
    :param bias:                Whether to use bias in the linear layer.``
    :param dropout:             The probability for the dropout layer.
    :param parametrization:     The parametrization to use. Defines the initialization
        and learning rate scaling for the parameters of the module.
    :param cov:                 Covariance structure of the weights.
    """

    def __init__(
        self,
        in_size: int,
        hidden_sizes: list[int],
        out_size: int,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
        inplace: bool | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: (
            params.FactorizedCovariance | list[params.FactorizedCovariance] | None
        ) = None,
    ) -> None:

        self.in_size = in_size
        self.hidden_sizes = hidden_sizes
        self.out_size = out_size
        params = {} if inplace is None else {"inplace": inplace}
        cov_list = []
        if isinstance(cov, list):
            cov_list = cov
            if len(cov_list) != len(hidden_sizes) + 1:
                raise ValueError(
                    "The length of the covariance list must be equal to the number of layers."
                )
        elif isinstance(cov, bnn.params.FactorizedCovariance):
            cov_list = [copy.deepcopy(cov) for _ in range(len(hidden_sizes) + 1)]
        else:
            cov_list = [None for _ in range(len(hidden_sizes) + 1)]

        # Layers
        layers = [
            bnn.Linear(
                in_features=self.in_size,
                out_features=self.hidden_sizes[0],
                bias=bias,
                layer_type="input",
                cov=cov_list[0],
            ),
        ]
        if norm_layer is not None:
            layers.append(norm_layer(self.hidden_sizes[0]))

        if activation_layer is not None:
            layers.append(activation_layer(**params))

        layers.append(nn.Dropout(dropout, **params))

        for idx_layer, (in_features, out_features) in enumerate(
            zip(self.hidden_sizes[0:-1], self.hidden_sizes[1:])
        ):
            layers.append(
                bnn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias,
                    cov=cov_list[idx_layer + 1],
                )
            )
            if norm_layer is not None:
                layers.append(norm_layer(out_features))
            if activation_layer is not None:
                layers.append(activation_layer(**params))

            layers.append(nn.Dropout(dropout, **params))

        layers.append(
            bnn.Linear(
                in_features=(
                    self.hidden_sizes[-1]
                    if len(self.hidden_sizes) > 0
                    else self.in_size
                ),
                out_features=self.out_size,
                bias=bias,
                layer_type="output",
                cov=cov_list[-1],
            )
        )
        layers.append(nn.Dropout(dropout, **params))

        super().__init__(*layers, parametrization=parametrization)
