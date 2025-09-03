import math

from numpy import testing as npt
import torch
from torch import nn

from inferno import bnn
from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "parametrization",
    [params.Standard, params.NeuralTangent, params.MaximalUpdate],
)
def test_changing_parametrization_of_bnn_module_changes_parametrization_of_children(
    parametrization,
):
    """When defining a BNN module with a certain parametrization, changing the parametrization
    of the module should change the parametrization of all children that are BNN modules.
    """

    class MyModel(bnn.BNNMixin, nn.Module):

        def __init__(self):
            super().__init__(parametrization=params.Standard())
            self.linear1 = bnn.Linear(
                2,
                5,
                layer_type="input",
                bias=True,
            )
            self.linear2 = bnn.Linear(
                5,
                3,
                bias=False,
                cov=params.FactorizedCovariance(),
            )
            self.linear3 = bnn.Linear(
                3,
                1,
                bias=True,
                cov=params.LowRankCovariance(rank=1),
            )
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()

            self.reset_parameters()

        def forward(self, input):
            input = self.linear1(input)
            input = self.relu1(input)
            input = self.linear2(input)
            input = self.relu2(input)
            input = self.linear3(input)
            input = self.relu3(input)
            return input

    model = MyModel()

    # Check initial parametrization
    for child in model.children():
        if isinstance(child, bnn.BNNMixin):
            assert isinstance(child.parametrization, params.Standard)

    # Change parametrization and check
    model.parametrization = parametrization()

    for child in model.children():
        if isinstance(child, bnn.BNNMixin):
            assert isinstance(child.parametrization, parametrization)
