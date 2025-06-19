from __future__ import annotations

import math

from .parametrization import LayerType, Parametrization, ParamType


class Standard(Parametrization):
    r"""Standard Parametrization (SP).

    The default parametrization for neural networks in PyTorch. Also known as the 'fan_in'
    or 'LeCun' initialization. 'Kaiming' initialization is the same up to multiplicative
    constants.
    """

    def weight_init_scale(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: LayerType = "hidden",
    ) -> float:
        return 1 / math.sqrt(fan_in)

    def bias_init_scale(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: LayerType = "hidden",
    ) -> float:
        # NOTE:  In Table 3 of [Tensor Programs V](http://arxiv.org/abs/2203.03466)
        # the bias in standard parametrization has scaling 1 / sqrt(fan_in) as
        # does the weight, but in the paper the fan_in of the bias is 1, rather
        # than the fan_in of the weight. This differs from the default initialization
        # in PyTorch which is 1.0 / math.sqrt(fan_in).
        return 1.0

    def weight_lr_scale(
        self,
        fan_in: int,
        fan_out: int,
        optimizer: str,
        layer_type: LayerType = "hidden",
    ) -> float:
        return 1.0

    def bias_lr_scale(
        self,
        fan_in,
        fan_out,
        optimizer: str,
        layer_type: LayerType = "hidden",
    ):
        return 1.0
