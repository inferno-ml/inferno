"""Loss functions."""

from .variational_free_energy import VariationalFreeEnergy

VariationalFreeEnergy.__module__ = "inferno.loss_fns"
NegativeELBO = VariationalFreeEnergy

__all__ = [
    "VariationalFreeEnergy",
    "NegativeELBO",
]
