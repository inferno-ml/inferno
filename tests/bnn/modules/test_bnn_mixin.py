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

        def reset_parameters(self):
            pass

    my_bnn_module = MyBNNModule(**kwargs)

    with pytest.raises(NotImplementedError):
        my_bnn_module(x)

    # Mixin as second superclass falls back to nn.Module.forward
    class MyBNNModule(TorchClass, bnn.BNNMixin):
        pass

        def reset_parameters(self):
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

        def reset_parameters(self):
            pass

    my_bnn_module = MyBNNModule(**kwargs, parametrization=parametrization)

    assert isinstance(my_bnn_module.parametrization, parametrization.__class__)


@pytest.mark.parametrize(
    "TorchClass,kwargs",
    [
        (nn.Linear, {}),
        (nn.Conv1d, {"kernel_size": 1}),
        (nn.Conv2d, {"kernel_size": 1}),
        (nn.Conv3d, {"kernel_size": 1}),
    ],
)
def test_mixin_prohibits_certain_torch_modules(TorchClass: nn.Module, kwargs: dict):

    # Check that reset_parameters and parameters_and_lrs fail when TorchClass used
    class MyBNNModule(bnn.BNNMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = TorchClass(3, 2, bias=True, **kwargs)

    my_bnn_module = MyBNNModule()

    with pytest.raises(NotImplementedError):
        my_bnn_module.reset_parameters()

    with pytest.raises(NotImplementedError):
        my_bnn_module.parameters_and_lrs(lr=1.0, optimizer="SGD")


@pytest.mark.parametrize(
    "TorchClass,kwargs",
    [
        (nn.LayerNorm, {"normalized_shape": (3, 2)}),
        (nn.GroupNorm, {"num_groups": 2, "num_channels": 2}),
        (nn.BatchNorm1d, {"num_features": 3}),
        (nn.BatchNorm2d, {"num_features": 3}),
        (nn.BatchNorm3d, {"num_features": 3}),
        (nn.Sequential, {}),
        (nn.ModuleList, {}),
        (nn.ModuleDict, {}),
    ],
)
def test_mixin_allows_certain_torch_modules(TorchClass: nn.Module, kwargs: dict):

    # Check that reset_parameters and parameters_and_lrs work when TorchClass used
    class MyBNNModule(bnn.BNNMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = TorchClass(**kwargs)

    my_bnn_module = MyBNNModule()
    my_bnn_module.reset_parameters()
    my_bnn_module.parameters_and_lrs(lr=1.0, optimizer="SGD")
