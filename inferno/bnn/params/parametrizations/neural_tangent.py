from __future__ import annotations

from .parametrization import LayerType, Parametrization, ParamType


class NeuralTangent(Parametrization):
    r"""Neural Tangent Parametrization (NTP).

    The neural tangent parametrization enables the theoretical analysis of the training dynamics of
    infinite-width neural networks (`Jacot et al., 2018`_) via the neural tangent kernel. However,
    NTP does not admit feature learning, as features are effectively fixed at initialization.

    .. _Jacot et al., 2018: http://arxiv.org/abs/1806.07572
    """

    def weight_init_scale(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: LayerType = "hidden",
    ) -> float:
        return 1.0

    def bias_init_scale(
        self,
        fan_in: int,
        fan_out: int,
        layer_type: LayerType = "hidden",
    ) -> float:
        return 1.0

    def weight_lr_scale(
        self,
        fan_in: int,
        fan_out: int,
        optimizer: str,
        layer_type: LayerType = "hidden",
    ) -> float:
        # TODO: double check this scaling
        if optimizer == "SGD":
            if layer_type == "input":
                return 1.0
            elif layer_type == "output":
                return 1.0 / fan_in
            else:
                return 1.0 / fan_in
        elif optimizer == "Adam":
            raise NotImplementedError(
                f"LR scaling not implemented for optimizer '{optimizer}'."
            )
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
        # TODO: double check this scaling
        return 1 / fan_in
