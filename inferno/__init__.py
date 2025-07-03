"""Bayesian Neural Networks in PyTorch."""

from importlib.metadata import PackageNotFoundError, version

from . import bnn, datasets, loss_fns, models

__all__ = ["bnn", "datasets", "loss_fns", "models"]

try:
    __version__ = version("inferno")
except PackageNotFoundError:
    # package is not installed
    pass
