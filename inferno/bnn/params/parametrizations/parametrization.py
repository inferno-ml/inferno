from __future__ import annotations

import abc
from typing import Literal

LayerType = Literal["input", "hidden", "output"]
ParamType = Literal["weight", "bias"]


class Parametrization(abc.ABC):
    """Abstract base class for all neural network parametrizations."""

    def weight_init_scale(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: LayerType = "hidden",
    ) -> float:
        """Compute the weight initialization scale for the given layer.

        :param fan_in: Number of inputs to the layer.
        :param fan_out: Number of outputs from the layer.
        :param layer_type: Type of the layer. Can be one of "input", "hidden", or "output".
        """
        raise NotImplementedError

    def bias_init_scale(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: LayerType = "hidden",
    ) -> float:
        """Compute the bias initialization scale for the given layer.

        :param fan_in: Number of inputs to the layer.
        :param fan_out: Number of outputs from the layer.
        :param layer_type: Type of the layer. Can be one of "input", "hidden", or "output".
        """
        raise NotImplementedError

    def weight_lr_scale(
        self,
        fan_in: int,
        fan_out: int,
        optimizer: str,
        layer_type: LayerType = "hidden",
    ) -> float:
        """Compute the learning rate scale for the weight parameters.

        :param fan_in: Number of inputs to the layer.
        :param fan_out: Number of outputs from the layer.
        :param optimizer: Optimizer being used.
        :param layer_type: Type of the layer. Can be one of "input", "hidden", or "output".
        """
        raise NotImplementedError

    def bias_lr_scale(
        self,
        fan_in: int,
        fan_out: int,
        optimizer: str,
        layer_type: LayerType = "hidden",
    ) -> float:
        """Compute the learning rate scale for the bias parameters.

        :param fan_in: Number of inputs to the layer.
        :param fan_out: Number of outputs from the layer.
        :param optimizer: Optimizer being used.
        :param layer_type: Type of the layer. Can be one of "input", "hidden", or "output".
        """
        raise NotImplementedError
