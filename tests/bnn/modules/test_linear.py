import copy
import math

import numpy as np
import numpy.testing as npt
import torch
from torch import nn

from inferno import bnn
from inferno.bnn import params

import pytest


@pytest.mark.parametrize("batch_shape", [(), (1,), (3,), (4, 1, 2)])
@pytest.mark.parametrize("in_features, out_features", [(5, 3), (3, 2)])
@pytest.mark.parametrize("cov", [None, params.FactorizedCovariance()])
def test_generalizes_pytorch_linear_layer(batch_shape, in_features, out_features, cov):
    """Test whether the BNN linear layer generalizes the PyTorch linear layer."""
    generator = torch.Generator().manual_seed(0)

    input = torch.randn(batch_shape + (in_features,), generator=generator)
    linear = torch.nn.Linear(in_features=in_features, out_features=out_features)
    linear.weight = torch.nn.Parameter(linear.weight)
    linear.bias = torch.nn.Parameter(linear.bias)

    bnn_linear = bnn.Linear(in_features=in_features, out_features=out_features, cov=cov)
    bnn_linear.params.weight = torch.nn.Parameter(linear.weight)
    bnn_linear.params.bias = torch.nn.Parameter(linear.bias)

    npt.assert_allclose(
        linear(input).detach().numpy(),
        bnn_linear(input, sample_shape=None, generator=generator).detach().numpy(),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "linear_layer_to_load",
    [
        bnn.Linear(3, 2),
        bnn.Linear(3, 2, cov=params.FactorizedCovariance()),
        nn.Linear(3, 2),
    ],
)
def test_load_from_state_dict(linear_layer_to_load):
    """Test whether the load_from_state_dict method is working for torch and inferno
    linear layers."""
    state_dict = linear_layer_to_load.state_dict()
    new_linear_layer = bnn.Linear(
        linear_layer_to_load.in_features,
        linear_layer_to_load.out_features,
        cov=params.FactorizedCovariance(),
    )
    new_linear_layer.load_state_dict(
        state_dict,
        strict=hasattr(linear_layer_to_load, "params.cov"),
    )

    prefix = "params." if isinstance(linear_layer_to_load, bnn.BNNMixin) else ""

    npt.assert_allclose(
        new_linear_layer.params.weight.detach().numpy(),
        state_dict[prefix + "weight"].detach().numpy(),
    )
    npt.assert_allclose(
        new_linear_layer.params.bias.detach().numpy(),
        state_dict[prefix + "bias"].detach().numpy(),
    )

    if hasattr(linear_layer_to_load, "params.cov"):
        npt.assert_allclose(
            new_linear_layer.params.cov_params.detach().numpy(),
            state_dict[prefix + "cov_params"].detach().numpy(),
        )


@pytest.mark.parametrize("seed", [0, 45234, 42])
def test_forward_is_deterministic_given_generator(seed):
    """Test whether the forward method is deterministic given a generator."""
    linear_layer = bnn.Linear(5, 3, cov=params.FactorizedCovariance())

    input = torch.randn(3, 5, generator=torch.Generator().manual_seed(seed + 2452345))
    output1 = linear_layer(input, generator=torch.Generator().manual_seed(seed))
    output2 = linear_layer(input, generator=torch.Generator().manual_seed(seed))

    npt.assert_allclose(output1.detach().numpy(), output2.detach().numpy())


@pytest.mark.parametrize("sample_shape", [(), (1,), (2,), (3, 2)])
@pytest.mark.parametrize("batch_shape", [(), (1,), (3,), (4, 1, 2)])
@pytest.mark.parametrize(
    "linear_layer",
    [
        bnn.Linear(5, 3, cov=params.FactorizedCovariance()),
        bnn.Linear(3, 2, bias=False, cov=params.LowRankCovariance(2)),
        bnn.Linear(3, 2, bias=False, cov=None),
    ],
)
def test_shape(sample_shape, batch_shape, linear_layer):
    """Test whether the output shape is correct."""
    generator = torch.Generator().manual_seed(0)
    input = torch.randn(*batch_shape, linear_layer.in_features, generator=generator)
    output = linear_layer(input, sample_shape=sample_shape, generator=generator)
    assert output.shape == (*sample_shape, *batch_shape, linear_layer.out_features)


def test_register_forward_hook():
    """Test whether the register_forward_hook method is working."""
    generator = torch.Generator().manual_seed(0)

    linear_layer = bnn.Linear(5, 3, cov=params.FactorizedCovariance())
    test_dict = {"hook_has_fired": False}

    def hook(module, input, output):
        test_dict["hook_has_fired"] = True

    linear_layer.register_forward_hook(hook)
    linear_layer(torch.randn(3, 5), generator=generator)

    assert test_dict["hook_has_fired"]


def random_fourier_features(
    x,
    num_features=100,
    outputscale=1.0,
    lengthscale=0.2,
    generator: torch.Generator | None = None,
):

    # Generate frequencies and phases
    frequencies = (
        torch.randn(x.shape[1], num_features, generator=generator) / lengthscale
    )
    phases = 2 * np.pi * torch.rand(num_features, generator=generator)

    # Compute random Fourier features
    projected = x @ frequencies + phases
    features = outputscale / np.sqrt(num_features) * torch.cos(projected)
    return features


@pytest.mark.parametrize(
    "momentum,nesterov",
    [
        (0.0, False),
        (0.9, False),
        (0.9, True),
    ],
)
@pytest.mark.parametrize(
    "cov",
    [params.LowRankCovariance(5), params.FactorizedCovariance()],
    ids=lambda x: x.__class__.__name__,
)
def test_linear_regression_nullspace_invariance(momentum, nesterov, cov):
    """
    Tests whether Bayesian linear regression with a factorized covariance leaves the prior
    invariant in the nullspace of the data.
    """
    torch.manual_seed(2452345)

    with torch.no_grad():
        # Data

        def feature_map(x):
            return random_fourier_features(x, num_features=100)

        def f(x):
            return torch.sin(x * 2 * np.pi)

        def y(x, noise_scale=0.3):
            return f(x).squeeze() + noise_scale * torch.randn(x.shape[0])

        num_train_data = 100

        X_train_raw = torch.concatenate(
            [
                2 * (torch.rand((num_train_data, 1)) - 0.5),
            ],
            dim=0,
        )
        X_train = feature_map(X_train_raw)
        y_train = y(X_train_raw)
        U, S, VT = torch.linalg.svd(X_train, full_matrices=True)
        V = VT.mT
        # V_range = V[:, : X_train.shape[0]]
        V_null = V[:, X_train.shape[0] :]

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

        # Dataloader
        batch_size = 64
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    # Model
    model = bnn.Linear(X_train.shape[-1], 1, bias=False, cov=cov)
    model.reset_parameters()
    prior_model = copy.deepcopy(model)

    # Loss function
    loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=1e-1,
        momentum=momentum,
        nesterov=nesterov,
    )

    # Training loop
    num_epochs = 20
    num_samples_train_loss = 32
    for epoch in range(num_epochs):
        model.train()

        for X_batch, y_batch in iter(train_dataloader):
            optimizer.zero_grad()

            # Prediction
            y_batch_pred = model(
                X_batch, sample_shape=(num_samples_train_loss,)
            ).squeeze(-1)

            # Loss
            loss = loss_fn(y_batch_pred, y_batch.expand(num_samples_train_loss, -1))

            # Optimizer step
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            prior_cov_params = torch.vstack(
                [
                    factor_param.view(-1, prior_model.params.cov.rank)
                    for factor_param in prior_model.params.cov.factor.values()
                    if factor_param is not None
                ]
            )
            cov_params = torch.vstack(
                [
                    factor_param.view(-1, model.params.cov.rank)
                    for factor_param in model.params.cov.factor.values()
                    if factor_param is not None
                ]
            )

            null_space_proj_mean = V_null.mT @ model.params.weight.ravel()
            null_space_proj_cov = V_null.mT @ cov_params @ cov_params.mT @ V_null

            null_space_proj_prior_mean = V_null.mT @ prior_model.params.weight.ravel()
            null_space_proj_prior_cov = (
                V_null.mT @ prior_cov_params @ prior_cov_params.mT @ V_null
            )

            npt.assert_allclose(
                null_space_proj_mean.numpy(),
                null_space_proj_prior_mean.numpy(),
                atol=1e-5,
                rtol=1e-5,
            )

            npt.assert_allclose(
                null_space_proj_cov.numpy(),
                null_space_proj_prior_cov.numpy(),
                atol=1e-5,
                rtol=1e-5,
            )


@pytest.mark.parametrize(
    "momentum,nesterov",
    [
        (0.0, False),
        (0.9, False),
        (0.9, True),
    ],
)
@pytest.mark.parametrize(
    "cov",
    [params.LowRankCovariance(5), params.FactorizedCovariance()],
    ids=lambda x: x.__class__.__name__,
)
def test_logistic_regression_nullspace_invariance(momentum, nesterov, cov):
    """
    Tests whether Bayesian logistic regression with a factorized covariance leaves the prior
    invariant in the nullspace of the data.
    """
    torch.manual_seed(2452345)

    with torch.no_grad():
        # Data

        def feature_map(x):
            return random_fourier_features(x, num_features=100)

        def y(x):
            return torch.as_tensor(
                (x @ torch.ones(input_dim)) >= 0, dtype=torch.float32
            )

        input_dim = 2
        num_train_data = 40

        X_train_raw = torch.concatenate(
            [
                0.4 * torch.randn((num_train_data // 2, input_dim))
                + torch.ones(input_dim),
                0.4 * torch.randn((num_train_data // 2, input_dim))
                - torch.ones(input_dim),
            ],
            axis=0,
        )
        X_train = feature_map(X_train_raw)
        y_train = y(X_train_raw)
        U, S, VT = torch.linalg.svd(X_train, full_matrices=True)
        V = VT.mT
        V_null = V[:, X_train.shape[0] :]

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)

        # Dataloader
        batch_size = 64
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    # Model
    model = bnn.Sequential(
        bnn.Linear(X_train.shape[-1], 1, bias=False, cov=cov),
        nn.Flatten(-2, -1),
        nn.Sigmoid(),
        parametrization=params.Standard(),
    )
    model[-3].reset_parameters()
    prior_model = copy.deepcopy(model)

    # Loss function
    loss_fn = nn.BCELoss()

    # Optimizer
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=1e0,
        momentum=momentum,
        nesterov=nesterov,
    )

    # Training loop
    num_epochs = 20
    num_samples_train_loss = 32
    for epoch in range(num_epochs):
        model.train()

        for X_batch, y_batch in iter(train_dataloader):
            optimizer.zero_grad()

            # Prediction
            y_batch_pred = model(X_batch, sample_shape=(num_samples_train_loss,))

            # Loss
            loss = loss_fn(y_batch_pred, y_batch.expand(num_samples_train_loss, -1))

            # Optimizer step
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            prior_cov_params = torch.vstack(
                [
                    factor_param.view(-1, prior_model[-3].params.cov.rank)
                    for factor_param in prior_model[-3].params.cov.factor.values()
                    if factor_param is not None
                ]
            )
            cov_params = torch.vstack(
                [
                    factor_param.view(-1, model[-3].params.cov.rank)
                    for factor_param in model[-3].params.cov.factor.values()
                    if factor_param is not None
                ]
            )

            null_space_proj_mean = V_null.mT @ model[-3].params.weight.ravel()
            null_space_proj_cov = V_null.mT @ cov_params @ cov_params.mT @ V_null

            null_space_proj_prior_mean = (
                V_null.mT @ prior_model[-3].params.weight.ravel()
            )
            null_space_proj_prior_cov = (
                V_null.mT @ prior_cov_params @ prior_cov_params.mT @ V_null
            )

            npt.assert_allclose(
                null_space_proj_mean.numpy(),
                null_space_proj_prior_mean.numpy(),
                atol=1e-5,
                rtol=1e-5,
            )

            npt.assert_allclose(
                null_space_proj_cov.numpy(),
                null_space_proj_prior_cov.numpy(),
                atol=1e-5,
                rtol=1e-5,
            )


@pytest.mark.parametrize(
    "cov",
    [
        params.FactorizedCovariance(),
        params.DiagonalCovariance(),
        params.LowRankCovariance(rank=2),
        params.KroneckerCovariance(input_rank=2, output_rank=3),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_if_bias_is_none_cov_bias_is_none(cov):
    """Test whether if the bias is None the covariance doesn't have bias parameters."""
    linear_layer = bnn.Linear(5, 3, bias=False, cov=cov)

    for name, param in linear_layer.params.cov.named_parameters():
        if "bias" in name:
            assert param is None


@pytest.mark.parametrize(
    "parametrization",
    [
        params.Standard(),
        params.NeuralTangent(),
        params.MaximalUpdate(),
    ],
    ids=lambda x: x.__class__.__name__,
)
@pytest.mark.parametrize("layer_type", ["input", "hidden", "output"], ids=lambda x: x)
def test_weight_initialization(parametrization, layer_type):
    """Test whether the initialization of the weights is correct."""
    torch.manual_seed(2452345)

    with torch.no_grad():
        linear_layer = bnn.Linear(
            333,
            333,
            layer_type=layer_type,
            parametrization=parametrization,
            cov=None,
        )

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
            linear_layer.params.weight
        )

        npt.assert_allclose(
            linear_layer.weight.mean(),
            0.0,
            atol=1e-2,
            rtol=1e-2,
        )

        npt.assert_allclose(
            linear_layer.weight.std(),
            parametrization.weight_init_scale(
                fan_in=fan_in, fan_out=fan_out, layer_type=layer_type
            ),
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.parametrize(
    "parametrization",
    [
        params.Standard(),
        params.NeuralTangent(),
        params.MaximalUpdate(),
    ],
    ids=lambda x: x.__class__.__name__,
)
@pytest.mark.parametrize("layer_type", ["input", "hidden", "output"], ids=lambda x: x)
def test_bias_initialization(parametrization, layer_type):
    """Test whether the initialization of the bias is correct."""
    torch.manual_seed(2452345)

    with torch.no_grad():
        linear_layer = bnn.Linear(
            333,
            333,
            bias=True,
            layer_type=layer_type,
            parametrization=parametrization,
            cov=None,
        )

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
            linear_layer.params.weight
        )

        npt.assert_allclose(
            linear_layer.bias.mean(),
            0.0,
            atol=1e-1,
            rtol=1e-1,
        )

        npt.assert_allclose(
            linear_layer.bias.std(),
            parametrization.bias_init_scale(
                fan_in=fan_in, fan_out=fan_out, layer_type=layer_type
            ),
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.parametrize(
    "parametrization",
    [
        params.Standard(),
        params.NeuralTangent(),
        params.MaximalUpdate(),
    ],
    ids=lambda x: x.__class__.__name__,
)
@pytest.mark.parametrize("layer_type", ["input", "hidden", "output"], ids=lambda x: x)
@pytest.mark.parametrize(
    "layer_width,cov",
    [
        (33, params.FactorizedCovariance()),
        (100, params.LowRankCovariance(5)),
        (100, params.KroneckerCovariance()),
        (100, params.DiagonalCovariance()),
    ],
    ids=lambda x: (
        x.__class__.__name__
        if isinstance(x, params.FactorizedCovariance)
        else f"width_{x}"
    ),
)
def test_covariance_parameter_initialization(
    parametrization, layer_type, layer_width, cov
):
    """Test whether the initialization of the covariance parameters is correct."""
    torch.manual_seed(2452345)

    with torch.no_grad():
        linear_layer = bnn.Linear(
            layer_width,
            layer_width,
            bias=True,
            layer_type=layer_type,
            parametrization=parametrization,
            cov=cov,
        )

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
            linear_layer.params.weight
        )

        if isinstance(cov, params.DiagonalCovariance):
            covariance_specific_scaling_factor = 1.0
        elif isinstance(cov, params.KroneckerCovariance):
            covariance_specific_scaling_factor = 1 / math.sqrt(cov.input_rank)
        elif isinstance(cov, params.LowRankCovariance):
            covariance_specific_scaling_factor = 1 / math.sqrt(cov.rank)
        elif isinstance(cov, params.FactorizedCovariance):
            covariance_specific_scaling_factor = 1 / math.sqrt(cov.rank)
        else:
            raise NotImplementedError()

        for name, param in linear_layer.params.cov.named_parameters():
            if param is not None:
                npt.assert_allclose(
                    param.mean(),
                    0.0,
                    atol=1e-1,
                    rtol=1e-1,
                )
            if "weight" in name:
                npt.assert_allclose(
                    param.std(),
                    parametrization.weight_init_scale(
                        fan_in=fan_in, fan_out=fan_out, layer_type=layer_type
                    )
                    * covariance_specific_scaling_factor,
                    atol=1e-1,
                    rtol=1e-1,
                )
            elif "bias" in name:
                npt.assert_allclose(
                    param.std(),
                    parametrization.bias_init_scale(
                        fan_in=fan_in, fan_out=fan_out, layer_type=layer_type
                    )
                    * covariance_specific_scaling_factor,
                    atol=1e-1,
                    rtol=1e-1,
                )
