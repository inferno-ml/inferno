import torch
from torch import nn

from inferno import bnn

import pytest


@pytest.mark.parametrize(
    "TorchClass,kwargs",
    [
        (nn.Linear, {"in_features": 5, "out_features": 2}),
        (nn.Conv1d, {"in_channels": 3, "out_channels": 1, "kernel_size": 1}),
    ],
)
def test_mixin_overrides_torch_module_forward(TorchClass: nn.Module, kwargs: dict):
    """Test when mixing in a BNNMixin into an nn.Module forces reimplementing forward."""

    x = torch.zeros((3, 5))

    # Mixin as first superclass forces reimplementation
    class MyBNNModule(bnn.BNNMixin, TorchClass):
        pass

    my_bnn_module = MyBNNModule(**kwargs)

    with pytest.raises(NotImplementedError):
        my_bnn_module(x)

    # Mixin as second superclass falls back to nn.Module.forward
    class MyBNNModule(TorchClass, bnn.BNNMixin):
        pass

    my_bnn_module = MyBNNModule(**kwargs)

    my_bnn_module(x)  # Does not raise error.


@pytest.mark.parametrize(
    "TorchClass,kwargs",
    [
        (nn.Linear, {"in_features": 5, "out_features": 2}),
        (nn.Conv1d, {"in_channels": 3, "out_channels": 1, "kernel_size": 1}),
    ],
)
@pytest.mark.parametrize(
    "parametrization",
    [bnn.params.SP(), bnn.params.MUP(), bnn.params.NTP()],
    ids=lambda c: c.__class__.__name__,
)
def test_mixin_allows_setting_parametrization(
    TorchClass: nn.Module, kwargs: dict, parametrization: bnn.params.Parametrization
):

    class MyBNNModule(bnn.BNNMixin, TorchClass):
        pass

    my_bnn_module = MyBNNModule(**kwargs, parametrization=parametrization)

    assert isinstance(my_bnn_module.parametrization, parametrization.__class__)
