import copy

from numpy import testing as npt
import torch
from torch import nn

from inferno import bnn
from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "model",
    [
        bnn.Sequential(
            bnn.Linear(5, 3, layer_type="input"),
            bnn.Linear(3, 2, cov=params.FactorizedCovariance()),
        ),
        bnn.Sequential(
            bnn.Linear(5, 3, cov=params.LowRankCovariance(2)),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.Softmax(dim=-1),
        ),
        bnn.Sequential(
            nn.Linear(5, 3),
        ),
        bnn.Sequential(
            nn.Linear(5, 6),
            nn.Flatten(0, 0),
            bnn.Linear(6, 1, cov=params.FactorizedCovariance(), layer_type="output"),
        ),
    ],
)
@pytest.mark.parametrize("sample_shape", [(), (1,), (2,), (3, 2)])
@pytest.mark.parametrize("batch_shape", [(1,), (3,)])
def test_shape(model, sample_shape, batch_shape):
    """Test whether the output shape of a container module is correct."""
    generator = torch.Generator().manual_seed(0)
    input = torch.randn(*batch_shape, model[0].in_features, generator=generator)
    output = model(input, sample_shape=sample_shape, generator=generator)
    assert output.shape == (
        *sample_shape,
        *batch_shape,
        model(torch.randn(model[0].in_features, generator=generator)).shape[-1],
    )


@pytest.mark.parametrize("seed", [0, 45234, 42])
def test_forward_is_deterministic_given_generator(seed):
    """Test whether the forward method is deterministic given a generator."""
    sequential_model = bnn.Sequential(
        bnn.Linear(4, 3),
        nn.ReLU(),
        bnn.Linear(3, 1, cov=params.FactorizedCovariance()),
    )

    input = torch.randn(3, 4, generator=torch.Generator().manual_seed(seed + 243534))
    output1 = sequential_model(input, generator=torch.Generator().manual_seed(seed))
    output2 = sequential_model(input, generator=torch.Generator().manual_seed(seed))

    npt.assert_allclose(output1.detach().numpy(), output2.detach().numpy())


def test_sample_shape_none_corresponds_to_forward_pass_with_mean_params():
    generator = torch.Generator().manual_seed(0)

    sequential_model = bnn.Sequential(
        bnn.Linear(4, 3, cov=params.FactorizedCovariance()),
        nn.ReLU(),
        bnn.Linear(3, 3, cov=params.FactorizedCovariance()),
        nn.ReLU(),
        bnn.Linear(3, 1, cov=params.FactorizedCovariance()),
    )
    sequential_model_deterministic = copy.deepcopy(sequential_model)
    for layer in sequential_model_deterministic:
        if hasattr(layer, "params"):
            layer.params.cov = None

    input = torch.randn(3, 4, generator=generator)

    output1 = sequential_model(input, sample_shape=None, generator=generator)
    output2 = sequential_model_deterministic(input, generator=generator)

    npt.assert_allclose(output1.detach().numpy(), output2.detach().numpy())


@pytest.mark.parametrize(
    "sequential_to_load,new_sequential,strict",
    [
        (
            nn.Sequential(
                nn.Linear(5, 3),
                nn.SiLU(),
                nn.Linear(3, 2),
            ),
            bnn.Sequential(
                bnn.Linear(5, 3, cov=params.FactorizedCovariance()),
                nn.SiLU(),
                nn.Linear(3, 2),
            ),
            False,
        ),
        (
            bnn.Sequential(
                bnn.Linear(5, 3, cov=params.FactorizedCovariance(), layer_type="input"),
                nn.ReLU(),
                bnn.Linear(
                    3, 2, cov=params.FactorizedCovariance(), layer_type="output"
                ),
                nn.Softmax(),
            ),
            bnn.Sequential(
                bnn.Linear(5, 3, cov=params.FactorizedCovariance(), layer_type="input"),
                nn.ReLU(),
                bnn.Linear(
                    3, 2, cov=params.FactorizedCovariance(), layer_type="output"
                ),
                nn.Softmax(),
            ),
            True,
        ),
    ],
)
def test_load_from_state_dict(sequential_to_load, new_sequential, strict):
    """Test whether the load_from_state_dict method is working for torch and inferno
    sequential layers."""
    torch.manual_seed(53124)

    missing_keys, unexpected_keys = new_sequential.load_state_dict(
        sequential_to_load.state_dict(),
        strict=strict,
    )

    assert len(unexpected_keys) == 0
    assert len(missing_keys) == 0 if strict else True


@pytest.mark.parametrize("layer_idx", [0, 1, 2, 3])
def test_register_forward_hook(layer_idx):
    """Test whether the register_forward_hook method is working on each layer of the
    sequential container."""
    model = bnn.Sequential(
        bnn.Linear(5, 3, cov=params.FactorizedCovariance()),
        nn.ReLU(),
        bnn.Linear(3, 2),
        nn.Softmax(dim=-1),
    )
    test_dict = {"hook_has_fired": False}

    def hook(module, input, output):
        test_dict["hook_has_fired"] = True

    model[layer_idx].register_forward_hook(hook)
    model(torch.randn(3, 5))

    assert test_dict["hook_has_fired"]


def test_setting_a_parametrization_overrides_module_parametrizations():
    """Test whether setting a parametrization on the sequential module overrides the
    parametrizations of the modules in the container."""
    model = bnn.Sequential(
        bnn.Linear(5, 3, cov=params.FactorizedCovariance()),
        nn.ReLU(),
        bnn.Linear(3, 2, parametrization=params.Standard()),
        parametrization=params.MaximalUpdate(),
    )

    for module in model.modules():
        if isinstance(module, bnn.BNNMixin):
            assert isinstance(module.parametrization, params.MaximalUpdate)

    model.parametrization = params.NeuralTangent()

    for module in model.modules():
        if isinstance(module, bnn.BNNMixin):
            assert isinstance(module.parametrization, params.NeuralTangent)


def test_no_parametrization_given():
    """Test whether modules retain their parametrization if no parametrization is given."""
    parametrizations = [
        params.Standard,
        params.MaximalUpdate,
    ]
    model = bnn.Sequential(
        bnn.Linear(
            5,
            3,
            cov=params.FactorizedCovariance(),
            parametrization=parametrizations[0](),
        ),
        bnn.Linear(3, 2, parametrization=parametrizations[1]()),
    )

    for i, module in enumerate(model):
        if isinstance(module, bnn.BNNMixin):
            assert isinstance(module.parametrization, parametrizations[i])


@pytest.mark.parametrize(
    "model",
    [
        bnn.Sequential(
            bnn.Linear(5, 3, cov=params.FactorizedCovariance()),
            nn.ReLU(),
            bnn.Linear(3, 2, cov=None),
            nn.ReLU(),
            bnn.Linear(2, 2, cov=params.FactorizedCovariance()),
            parametrization=params.MUP(),
        ),
    ],
)
@pytest.mark.parametrize(
    "idx",
    [
        slice(0, -1),
        slice(0, 1),
        0,
        -1,
    ],
)
def test_indexing_or_slicing_does_not_reset_parameters(model, idx):

    # Get parameters before slicing
    params_before = {name: param.clone() for name, param in model.named_parameters()}

    # Slice the model
    sliced_model = model[idx]

    # Get parameters after slicing
    params_after = {
        name: param.clone() for name, param in sliced_model.named_parameters()
    }

    # Verify parameters haven't changed
    for name in params_before:
        if name in params_after:
            torch.testing.assert_close(
                params_before[name],
                params_after[name],
            )
