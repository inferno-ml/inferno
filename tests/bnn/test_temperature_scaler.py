import copy

import numpy.testing as npt
import torch
from torch import nn

from inferno import bnn
from inferno.bnn import params

import pytest

model_loss_fn_input_dim_num_classes_tuples = [
    (
        bnn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            bnn.Linear(
                10,
                2,
                layer_type="output",
            ),
        ),
        nn.CrossEntropyLoss(),
        10,
        2,
    ),
    (
        bnn.Sequential(
            bnn.Linear(
                10,
                10,
                cov=params.FactorizedCovariance(),
                layer_type="input",
            ),
            nn.ReLU(),
            bnn.Linear(
                10,
                3,
                cov=params.FactorizedCovariance(),
                layer_type="output",
            ),
        ),
        nn.CrossEntropyLoss(),
        10,
        3,
    ),
    (
        bnn.Sequential(
            bnn.Linear(
                10,
                10,
                cov=params.FactorizedCovariance(),
                layer_type="input",
            ),
            nn.ReLU(),
            bnn.Linear(
                10,
                1,
                cov=params.FactorizedCovariance(),
                layer_type="output",
            ),
            nn.Flatten(-2, -1),
        ),
        nn.BCEWithLogitsLoss(),
        10,
        2,
    ),
]


@pytest.mark.parametrize(
    "model,loss_fn,input_dim,num_classes",
    model_loss_fn_input_dim_num_classes_tuples,
    ids=["deterministic", "probabilistic", "bce_logits"],
)
def test_temperature_scaler_on_different_models(model, loss_fn, input_dim, num_classes):

    torch.manual_seed(0)

    # Create a temperature scaler
    temperature_scaler = bnn.TemperatureScaler(loss_fn=loss_fn)

    # Create a dummy dataloader
    num_data = 50
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(num_data, input_dim),
            torch.randint(
                0,
                num_classes,
                (num_data,),
                dtype=(
                    torch.float32
                    if isinstance(loss_fn, nn.BCEWithLogitsLoss)
                    else torch.long
                ),
            ),
        ),
        batch_size=10,
    )

    temperature_param = None
    for name, param in model.named_parameters():
        if "temperature" in name:
            temperature_param = param
            break

    assert temperature_param.item() == 1.0

    # Optimize the temperature
    temperature_scaler.optimize(model, dataloader)

    assert temperature_param.item() != 1.0

    for name, param in model.named_parameters():
        if "temperature" in name:
            assert not param.requires_grad
        else:
            assert param.requires_grad


@pytest.mark.parametrize(
    "model,loss_fn,input_dim,num_classes",
    model_loss_fn_input_dim_num_classes_tuples,
    ids=["deterministic", "probabilistic", "bce_logits"],
)
def test_temperature_scaler_doesnt_modify_model_parameters(
    model, loss_fn, input_dim, num_classes
):
    torch.manual_seed(0)

    # Create a temperature scaler
    temperature_scaler = bnn.TemperatureScaler(loss_fn=loss_fn)

    # Create a dummy dataloader
    num_data = 50
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(num_data, input_dim),
            torch.randint(
                0,
                num_classes,
                (num_data,),
                dtype=(
                    torch.float32
                    if isinstance(loss_fn, nn.BCEWithLogitsLoss)
                    else torch.long
                ),
            ),
        ),
        batch_size=10,
    )

    temperature_param = None
    model_params_copy = {}
    for name, param in model.named_parameters():
        if "temperature" in name:
            temperature_param = param
        model_params_copy[name] = copy.deepcopy(param)

    # Optimize the temperature
    temperature_scaler.optimize(model, dataloader)

    # Check that the model parameters are not modified
    for name, param in model.named_parameters():
        if "temperature" not in name:
            npt.assert_array_equal(
                model_params_copy[name].detach().cpu().numpy(),
                param.detach().cpu().numpy(),
            )


@pytest.mark.parametrize(
    "model,loss_fn,input_dim,num_classes",
    model_loss_fn_input_dim_num_classes_tuples,
    ids=["deterministic", "probabilistic", "bce_logits"],
)
@pytest.mark.parametrize("training_mode", [True, False])
def test_temperature_scaling_works_regardless_of_training_or_eval_mode(
    model, loss_fn, input_dim, num_classes, training_mode
):
    torch.manual_seed(0)

    # Create a temperature scaler
    temperature_scaler = bnn.TemperatureScaler(loss_fn=loss_fn)

    # Create a dummy dataloader
    num_data = 50
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(num_data, input_dim),
            torch.randint(
                0,
                num_classes,
                (num_data,),
                dtype=(
                    torch.float32
                    if isinstance(loss_fn, nn.BCEWithLogitsLoss)
                    else torch.long
                ),
            ),
        ),
        batch_size=10,
    )

    # Set the model to training or evaluation mode
    if training_mode:
        model.train()
    else:
        model.eval()

    # Optimize the temperature
    temperature_scaler.optimize(model, dataloader)

    assert model.training == training_mode
