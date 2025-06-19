from __future__ import annotations

import copy
from typing import Callable

from torch import nn

from inferno import bnn
from inferno.bnn import params


class LeNet5(bnn.Sequential):
    """A simple convolutional neural network for image classification of 28x28 grayscale images.

    :param out_size:            Size of the output (i.e. number of classes).
    :param parametrization:     The parametrization to use. Defines the initialization
        and learning rate scaling for the parameters of the module.
    :param cov:                 Covariance structure of the weights.
    :param activation_layer:    Activation function following a linear layer.
    """

    def __init__(
        self,
        out_size: int = 10,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: params.FactorizedCovariance | None = None,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
    ) -> None:

        self.out_size = out_size

        # Layers
        layers = [
            bnn.Conv2d(
                1,
                6,
                kernel_size=5,
                padding=2,
                cov=(
                    copy.deepcopy(cov)
                    if isinstance(cov, params.DiagonalCovariance)
                    else copy.deepcopy(cov)
                ),
                layer_type="input",
            ),
            activation_layer() if activation_layer is not None else nn.Identity(),
            nn.MaxPool2d(kernel_size=2),
            bnn.Conv2d(
                6,
                16,
                kernel_size=5,
                cov=(
                    copy.deepcopy(cov)
                    if isinstance(cov, params.DiagonalCovariance)
                    else None
                ),
            ),
            activation_layer() if activation_layer is not None else nn.Identity(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            bnn.Linear(
                16 * 5 * 5,
                120,
                cov=(
                    copy.deepcopy(cov)
                    if isinstance(cov, params.DiagonalCovariance)
                    else None
                ),
            ),
            activation_layer() if activation_layer is not None else nn.Identity(),
            bnn.Linear(
                120,
                84,
                cov=(
                    copy.deepcopy(cov)
                    if isinstance(cov, params.DiagonalCovariance)
                    else None
                ),
            ),
            activation_layer() if activation_layer is not None else nn.Identity(),
            bnn.Linear(
                84,
                out_size,
                cov=(
                    copy.deepcopy(cov)
                    if isinstance(cov, params.DiagonalCovariance)
                    else copy.deepcopy(cov)
                ),
                layer_type="output",
            ),
        ]

        super().__init__(*layers, parametrization=parametrization)
