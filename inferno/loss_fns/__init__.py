"""Loss functions."""

from .focal_loss import FocalLoss
from .variational_free_energy import VariationalFreeEnergy
from .wrapped_torch_loss_fns import (
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    L1Loss,
    MSELoss,
    NLLLoss,
    _num_extra_dims,
    _predictions_and_expanded_targets,
)

VariationalFreeEnergy.__module__ = "inferno.loss_fns"
NegativeELBO = VariationalFreeEnergy

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "FocalLoss",
    "MultipleBatchDimensionsLossMixin",
    "L1Loss",
    "MSELoss",
    "NLLLoss",
    "NegativeELBO",
    "VariationalFreeEnergy",
    "_num_extra_dims",
    "_predictions_and_expanded_targets",
]
