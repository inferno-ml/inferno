"""Pre-defined models."""

from torch import nn

from .. import bnn
from .ensemble import Ensemble
from .lenet5 import LeNet5
from .mlp import MLP
from .resnet import (
    ResNet,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNeXt50_32X4D,
    ResNeXt101_32X8D,
    ResNeXt101_64X4D,
    WideResNet50,
    WideResNet101,
)

___all__ = [
    "Ensemble",
    "LeNet5",
    "MLP",
    "ResNet",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNeXt50_32X4D",
    "ResNeXt101_32X8D",
    "ResNeXt101_64X4D",
    "WideResNet50",
    "WideResNet101",
    "as_torch_model",
]
