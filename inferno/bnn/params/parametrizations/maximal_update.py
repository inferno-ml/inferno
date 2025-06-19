from __future__ import annotations

import math

from .parametrization import LayerType, Parametrization, ParamType


class MaximalUpdate(Parametrization):
    r"""Maximal update parametrization ($\mu P$).

    The maximal update parametrization (`Yang et al., 2021`_, `Yang et al., 2021b`_) enforces:
        - *stable training dynamics*, meaning (pre)activations and logits at activations are independent
          of width at initialization and features and logits do not explode during training.
        - *feature learning*, meaning the model can learn representations from the data at any width.

    This parametrization is particularly useful when training large models, since it maintains
    stable training dynamics at any width and enables hyperparameter transfer. This means one can
    tune the learning rate on a smaller model and achieve good generalization with the tuned
    learning rate for a large model.

    .. _Yang et al., 2021: http://arxiv.org/abs/2011.14522
    .. _Yang et al., 2021b: http://arxiv.org/abs/2203.03466
    """

    def weight_init_scale(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: LayerType = "hidden",
    ) -> float:
        if layer_type == "input":
            return 1.0 / math.sqrt(fan_in)
        elif layer_type == "output":
            return 1.0 / fan_in
        else:
            return 1.0 / math.sqrt(fan_in)

    def bias_init_scale(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: LayerType = "hidden",
    ) -> float:
        # NOTE: In Table 3 of [Tensor Programs V](http://arxiv.org/abs/2203.03466),
        # the bias has scaling 1.0 / sqrt(fan_in) as does the weight, but in the paper the
        # fan_in of the bias is 1, rather than the fan_in of the weight.
        return 1.0

    def weight_lr_scale(
        self,
        fan_in: int,
        fan_out: int,
        optimizer: str,
        layer_type: LayerType = "hidden",
    ) -> float:

        if optimizer == "SGD":
            if layer_type == "input":
                return fan_out
            elif layer_type == "output":
                return 1.0 / fan_in
            else:
                return 1.0
        elif optimizer == "Adam":
            if layer_type == "input":
                return 1.0
            elif layer_type == "output":
                return 1.0 / fan_in
            else:
                return 1.0 / fan_in
        else:
            raise NotImplementedError(
                f"LR scaling not implemented for optimizer '{optimizer}'."
            )

    def bias_lr_scale(
        self,
        fan_in: int,
        fan_out: int,
        optimizer: str,
        layer_type: LayerType = "hidden",
    ) -> float:
        if optimizer == "SGD":
            return fan_out
        elif optimizer == "Adam":
            return 1.0
        else:
            raise NotImplementedError(
                f"LR scaling not implemented for optimizer '{optimizer}'."
            )
