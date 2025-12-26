import torch
from torch import nn, testing

from inferno import bnn, loss_fns, models

import pytest


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize(
    "loss_fn,loss_fn_variance_reduced,model,input,target",
    [
        (
            loss_fns.MSELoss,
            loss_fns.MSELossVR,
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8, 8],
                out_size=1,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    None,
                    bnn.params.LowRankCovariance(4),
                ],
            ),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(4958)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(4958)),
        ),
        (
            loss_fns.MSELoss,
            loss_fns.MSELossVR,
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8, 8],
                out_size=1,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    None,
                    None,
                ],
            ),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(4958)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(4958)),
        ),
        (
            loss_fns.MSELoss,
            loss_fns.MSELossVR,
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8],
                out_size=1,
                bias=False,
                cov=[
                    None,
                    None,
                    bnn.params.FactorizedCovariance(),
                ],
            ),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(4958)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(4958)),
        ),
        (
            loss_fns.MSELoss,
            loss_fns.MSELossVR,
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8],
                out_size=1,
                bias=True,
                cov=[
                    None,
                    None,
                    bnn.params.FactorizedCovariance(),
                ],
            ),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(4958)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(4958)),
        ),
        (
            loss_fns.CrossEntropyLoss,
            loss_fns.CrossEntropyLossVR,
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=5,
                bias=True,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    bnn.params.LowRankCovariance(4),
                ],
            ),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.CrossEntropyLoss,
            loss_fns.CrossEntropyLossVR,
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=5,
                bias=False,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    bnn.params.LowRankCovariance(4),
                ],
            ),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.BCEWithLogitsLoss,
            loss_fns.BCEWithLogitsLossVR,
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=(),
                cov=None,
            ),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32).random_(2, generator=torch.Generator().manual_seed(3244)),
        ),
        (
            loss_fns.BCEWithLogitsLoss,
            loss_fns.BCEWithLogitsLossVR,
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=(),
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    bnn.params.LowRankCovariance(4),
                ],
            ),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32).random_(2, generator=torch.Generator().manual_seed(3244)),
        ),
    ],
)
def test_equals_expected_loss(
    loss_fn, loss_fn_variance_reduced, model, input, target, reduction
):

    # Necessary to avoid flaky tests since model initalization is not deterministic.
    with torch.random.fork_rng():  # Do not change global rng state.
        torch.manual_seed(23)
        model.reset_parameters()

        # Get representation of input and output layer
        model_representation = model[0:-1]
        output_layer = model[-1]

        # Evaluate loss functions
        num_samples = 10000
        loss = loss_fn(reduction=reduction)(
            model(
                input,
                sample_shape=(num_samples,),
                generator=torch.Generator().manual_seed(999),
            ),
            target,
        )
        loss_variance_reduced = loss_fn_variance_reduced(reduction=reduction)(
            model_representation(
                input,
                sample_shape=(num_samples,),
                generator=torch.Generator().manual_seed(999),
            ),
            output_layer,
            target,
        )

        if isinstance(
            loss_fn_variance_reduced(reduction=reduction), loss_fns.MSELossVR
        ):
            testing.assert_close(
                loss_variance_reduced,
                loss,
                atol=1e-2,
                rtol=1e-2,
            )
        elif isinstance(
            loss_fn_variance_reduced(reduction=reduction),
            (loss_fns.BCEWithLogitsLossVR, loss_fns.CrossEntropyLossVR),
        ):
            assert torch.all(loss <= loss_variance_reduced)
        else:
            raise NotImplementedError


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize(
    "loss_fn,loss_fn_variance_reduced,model,input,target",
    [
        (
            nn.MSELoss,
            loss_fns.MSELossVR,
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8],
                out_size=1,
                cov=None,
            ),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(8932)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(8932)),
        ),
        (
            nn.CrossEntropyLoss,
            loss_fns.CrossEntropyLossVR,
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=5,
                bias=False,
                cov=None,
            ),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            nn.BCEWithLogitsLoss,
            loss_fns.BCEWithLogitsLossVR,
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=(),
                cov=None,
            ),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32).random_(2, generator=torch.Generator().manual_seed(3244)),
        ),
    ],
)
def test_equals_torch_loss_for_deterministic_models(
    loss_fn, loss_fn_variance_reduced, model, input, target, reduction
):

    # Get representation of input and output layer
    model_representation = model[0:-1]
    output_layer = model[-1]

    # Compare losses
    loss = loss_fn(reduction=reduction)(
        model(input, sample_shape=None),
        target,
    )
    loss_variance_reduced = loss_fn_variance_reduced(reduction=reduction)(
        model_representation(input, sample_shape=None),
        output_layer,
        target,
    )

    testing.assert_close(
        loss_variance_reduced,
        loss,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "loss_fn_variance_reduced,model,sample_shape,input,target",
    [
        (
            loss_fns.MSELossVR(reduction="none"),
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8, 8],
                out_size=1,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    None,
                    bnn.params.LowRankCovariance(4),
                ],
            ),
            (6, 4),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(4958)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(4958)),
        ),
        (
            loss_fns.MSELossVR(reduction="none"),
            models.MLP(
                in_size=3,
                hidden_sizes=[8, 8, 8],
                out_size=1,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    None,
                    bnn.params.LowRankCovariance(4),
                ],
            ),
            (32,),
            torch.randn(64, 3, generator=torch.Generator().manual_seed(4958)),
            torch.randn(64, 1, generator=torch.Generator().manual_seed(4958)),
        ),
        (
            loss_fns.CrossEntropyLossVR(reduction="none"),
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=5,
                bias=True,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    bnn.params.LowRankCovariance(4),
                ],
            ),
            (
                5,
                6,
            ),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.CrossEntropyLossVR(reduction="none"),
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=5,
                bias=True,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    bnn.params.LowRankCovariance(4),
                ],
            ),
            (5,),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.CrossEntropyLossVR(reduction="none"),
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=5,
                bias=False,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    bnn.params.LowRankCovariance(4),
                ],
            ),
            (),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32, dtype=torch.long).random_(
                5, generator=torch.Generator().manual_seed(3244)
            ),
        ),
        (
            loss_fns.BCEWithLogitsLossVR(reduction="none"),
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=1,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    bnn.params.LowRankCovariance(3),
                ],
            ),
            (),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32).random_(2, generator=torch.Generator().manual_seed(3244)),
        ),
        (
            loss_fns.BCEWithLogitsLossVR(reduction="none"),
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=1,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    bnn.params.LowRankCovariance(3),
                ],
            ),
            (10,),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32).random_(2, generator=torch.Generator().manual_seed(3244)),
        ),
        (
            loss_fns.BCEWithLogitsLossVR(reduction="none"),
            models.MLP(
                in_size=6,
                hidden_sizes=[8, 8],
                out_size=1,
                cov=[
                    bnn.params.FactorizedCovariance(),
                    None,
                    bnn.params.LowRankCovariance(3),
                ],
            ),
            (
                10,
                32,
            ),
            torch.randn((32, 6), generator=torch.Generator().manual_seed(42)),
            torch.empty(32).random_(2, generator=torch.Generator().manual_seed(3244)),
        ),
    ],
)
def test_shape_for_no_reduction(
    loss_fn_variance_reduced,
    model,
    sample_shape,
    input,
    target,
):

    # Get representation of input and output layer
    model_representation = model[0:-1]
    output_layer = model[-1]

    loss_variance_reduced = loss_fn_variance_reduced(
        model_representation(input, sample_shape=sample_shape),
        output_layer,
        target,
    )

    assert loss_variance_reduced.shape == sample_shape + target.shape
