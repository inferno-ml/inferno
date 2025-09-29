import copy
import math

import numpy as np
import numpy.testing as npt
import torch
from torch import nn

from inferno import bnn
from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "layer_to_load",
    [
        # bnn.Linear(3, 2),
        # bnn.Linear(3, 2, cov=params.FactorizedCovariance()),
        # nn.Linear(3, 2),
    ],
)
def test_load_from_state_dict(layer_to_load):
    """Test whether the load_from_state_dict method is working for torch and inferno
    layers."""
    raise NotImplementedError


def test_generalizes_pytorch_multi_head_attention_layer(
    batch_shape, in_features, out_features
):
    """Test whether the BNN attention layer generalizes the PyTorch attention layer."""
    raise NotImplementedError
