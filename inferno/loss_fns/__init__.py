"""Loss functions."""

from .variational_free_energy import VariationalFreeEnergy
from .wrapped_torch_loss_fns import (
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    L1Loss,
    MSELoss,
    NLLLoss,
    inputs_and_expanded_targets,
)

VariationalFreeEnergy.__module__ = "inferno.loss_fns"
NegativeELBO = VariationalFreeEnergy

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "L1Loss",
    "MSELoss",
    "NLLLoss",
    "NegativeELBO",
    "VariationalFreeEnergy",
    "inputs_and_expanded_targets",
]
