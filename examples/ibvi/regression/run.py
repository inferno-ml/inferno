import pathlib

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torchmetrics
import tqdm

import inferno


def toy_regression(
    num_train_data: int = 100,
    num_test_data: int = 200,
    batch_size: int = 16,
    num_hidden_layers: int = 3,
    hidden_width: int = 16,
    num_epochs: int = 500,
    lr: float = 1e-2,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else torch.get_default_device()
    ),
    seed: int = 42,
    plot_data: bool = True,
    plot_learning_curves: bool = True,
    plot_prediction: bool = True,
    outdir: str = "plots",
):

    # Setup
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.manual_seed(seed)

    # Create output directory
    if plot_data or plot_learning_curves or plot_prediction:
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    # Generate training and test data
    with torch.no_grad():

        # Latent function
        def f(x):
            return torch.sigmoid(10 * x) - 0.5

        def y(x, noise_scale=0.02):
            return f(x).squeeze() + noise_scale * torch.randn(x.shape[0])

        # Training and test data
        X_train = torch.concatenate(
            [
                0.35 * torch.rand((num_train_data // 3, 1)) - 1.0,
                0.25 * torch.rand((num_train_data // 3, 1)) + 0.25,
                0.25 * torch.rand((num_train_data // 3, 1)) + 0.75,
            ],
            dim=0,
        )

        y_train = y(X_train)

        X_test = 3 * (torch.rand((num_test_data, 1)) - 0.5)
        y_test = y(X_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=False,
    )

    # Plot training and test data
    if plot_data:
        with torch.no_grad():
            fig, axs = plt.subplots(
                nrows=1,
                ncols=1,
                sharex="row",
                sharey="row",
                squeeze=False,
                figsize=(6, 2),
            )

            # Training data
            axs[0, 0].scatter(
                train_dataset[:][0].detach().cpu().numpy(),
                train_dataset[:][1].detach().cpu().numpy(),
                label=f"Training data",
                color="black",
                zorder=-1,
            )

            # Test data
            axs[0, 0].scatter(
                test_dataset[:][0].detach().cpu().numpy(),
                test_dataset[:][1].detach().cpu().numpy(),
                label=f"Test data",
                color="gray",
                alpha=0.1,
                zorder=-2,
            )

            # Axes
            # axs[0, -1].set(ylim=[-2.5, 2.5])

            # Legend
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(
                handles=handles,
                labels=labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                fancybox=False,
                shadow=False,
                ncol=3,
                frameon=False,
            )

            fig.tight_layout()
            plt.savefig(
                pathlib.Path(outdir) / "data.svg",
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close(fig)

    # --8<-- [start:model]
    from torch import nn

    from inferno import bnn, loss_fns

    model = bnn.Sequential(
        inferno.models.MLP(
            in_size=1,
            hidden_sizes=[hidden_width] * num_hidden_layers,
            out_size=1,
            activation_layer=nn.SiLU,  # (1)!
            cov=[bnn.params.FactorizedCovariance()]
            + [None] * (num_hidden_layers - 1)
            + [bnn.params.FactorizedCovariance()],
            bias=True,
        ),
        nn.Flatten(-2, -1),
        parametrization=bnn.params.MUP(),
    )
    # --8<-- [end:model]

    results_list = []

    # --8<-- [start:training]
    # Loss function
    loss_fn = loss_fns.MSELoss()

    # Optimizer
    optimizer = torch.optim.SGD(
        params=model.parameters_and_lrs(
            lr=lr, optimizer="SGD"
        ),  # Sets module-specific learning rates
        lr=lr,
        momentum=0.9,
    )

    # Training loop
    for epoch in tqdm.trange(num_epochs):
        model.train()

        for X_batch, y_batch in iter(train_dataloader):
            optimizer.zero_grad()

            X_batch = X_batch.to(device=device)
            y_batch = y_batch.to(device=device)

            y_batch_pred = model(X_batch)  # Uses a single parameter sample

            loss = loss_fn(y_batch_pred, y_batch)

            loss.backward()
            optimizer.step()
            # --8<-- [end:training]

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

        with torch.no_grad():
            # Model evaluation
            model.eval()

            test_metric_mse = torchmetrics.MeanSquaredError(squared=True, num_outputs=1)

            for X_test, y_test in iter(test_dataloader):

                # Prediction
                y_test_pred = model(X_test, sample_shape=(16,))

                # Metrics
                test_metric_mse.update(
                    (
                        y_test_pred.mean(dim=0)
                        if isinstance(model, bnn.BNNMixin)
                        else y_test_pred
                    ),
                    y_test,
                )

            results_list.append(
                {
                    "Dataset": train_dataset.__class__.__name__,
                    "Model": "IBVI MLP",
                    "Number of Parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    "Optimizer": optimizer.__class__.__name__,
                    "LR": lr,
                    "Epoch": epoch,
                    "Loss": loss_fn.__class__.__name__,
                    "Train Loss": loss.item(),
                    "Test MSE": test_metric_mse.compute().item(),
                }
            )
    results_df = pd.DataFrame(results_list)

    if plot_learning_curves:
        # Plot learning curves
        with torch.no_grad():
            fig, axs = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(7, 3),
            )

            # Train Loss
            sns.lineplot(
                data=results_df,
                x="Epoch",
                y="Train Loss",
                hue="Model",
                ax=axs[0],
                alpha=0.8,
                legend=True,
            )

            # Test MSE
            sns.lineplot(
                data=results_df,
                x="Epoch",
                y="Test MSE",
                hue="Model",
                ax=axs[1],
                alpha=0.8,
                legend=False,
            )

            # Axes
            axs[0].set(yscale="log")
            axs[1].set(yscale="log")

            # Legend
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(
                handles=handles,
                labels=labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                fancybox=False,
                shadow=False,
                ncol=len(labels),
                frameon=False,
            )
            axs[0].get_legend().remove()

            fig.align_labels()
            plt.tight_layout()

            plt.savefig(
                pathlib.Path(outdir) / "learning_curves.svg",
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close()

    if plot_prediction:
        # Plot predictions
        with torch.no_grad():
            fig, axs = plt.subplots(
                nrows=1,
                ncols=1,
                sharex="row",
                sharey="row",
                figsize=(7, 3),
            )

            # Inputs to predict at
            x_axis_width_factor = 2.0
            xs = torch.linspace(
                torch.min(X_train) * x_axis_width_factor,
                torch.max(X_train) * x_axis_width_factor,
                1000,
                device=device,
                dtype=dtype,
            ).unsqueeze(-1)

            # Latent function
            axs.plot(
                xs.detach().cpu().numpy(),
                f(xs).detach().cpu().numpy(),
                label=f"Latent function",
                color="black",
                linestyle="--",
                alpha=0.3,
            )

            # Training data
            axs.scatter(
                train_dataset[:][0].detach().cpu().numpy(),
                train_dataset[:][1].detach().cpu().numpy(),
                label=f"Training data",
                color="black",
                zorder=-1,
            )

            # Test data
            axs.scatter(
                test_dataset[:][0].detach().cpu().numpy(),
                test_dataset[:][1].detach().cpu().numpy(),
                label=f"Test data",
                color="gray",
                alpha=0.1,
                zorder=-2,
            )

            # Push-forward of posterior distribution
            model.eval()
            fs_pred = model(xs, sample_shape=(10000,))

            axs.plot(
                xs.detach().cpu().numpy(),
                fs_pred.nanmedian(dim=0).values.detach().cpu().numpy(),
                label=f"Prediction",
                color=f"C{0}",
                zorder=5,
            )
            axs.fill_between(
                xs.flatten().detach().cpu().numpy(),
                torch.quantile(fs_pred, q=0.025, dim=0),
                torch.quantile(fs_pred, q=0.975, dim=0),
                alpha=0.4,
                label=f"Uncertainty",
                color=f"C{0}",
            )

            # Samples
            axs.plot(
                xs.detach().cpu().numpy(),
                fs_pred[0:20].squeeze(-1).mT.detach().cpu().numpy(),
                alpha=0.3,
                linewidth=0.5,
                color=f"C{0}",
            )

            # Legend
            handles, labels = axs.get_legend_handles_labels()
            fig.legend(
                handles=handles,
                labels=labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                fancybox=False,
                shadow=False,
                ncol=5,
                frameon=False,
            )

            fig.align_labels()
            plt.tight_layout()

            plt.savefig(
                pathlib.Path(outdir) / "predictions.svg",
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close()


if __name__ == "__main__":
    fire.Fire(toy_regression)
