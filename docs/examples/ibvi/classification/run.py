import pathlib

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch
import torchmetrics
import tqdm

import inferno


def toy_classification(
    num_train_data: int = 200,
    num_test_data: int = 400,
    batch_size: int = 32,
    num_hidden_layers: int = 3,
    hidden_width: int = 64,
    num_epochs: int = 200,
    lr: float = 1e-2,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else torch.get_default_device()
    ),
    seed: int = 42,
    plot_data: bool = True,
    plot_learning_curves: bool = True,
    plot_decision_boundary: bool = True,
    outdir: str = "plots",
):

    # Setup
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.manual_seed(seed)

    # Create output directory
    if plot_data or plot_learning_curves or plot_decision_boundary:
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    # Generate training and test data
    X, y = make_moons(
        n_samples=num_train_data + num_test_data, noise=0.1, random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=num_test_data / (num_train_data + num_test_data),
        random_state=seed,
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(X_train, dtype=dtype), torch.as_tensor(y_train, dtype=dtype)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(X_test, dtype=dtype), torch.as_tensor(y_test, dtype=dtype)
    )

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
            fig, ax = plt.subplots(figsize=(4, 4))

            # Training data
            ax.scatter(
                X_train[:, 0],
                X_train[:, 1],
                c=y_train,
                cmap="RdBu",
                edgecolor="k",
                s=30,
                label="Training data",
            )

            # Test data
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap="RdBu",
                alpha=0.3,
                s=20,
                label="Test data",
            )

            ax.set_aspect("equal")

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(
                handles=handles,
                labels=labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.05),
                fancybox=False,
                shadow=False,
                ncol=3,
                frameon=False,
            )

            plt.tight_layout()
            plt.savefig(
                pathlib.Path(outdir) / "data.svg", bbox_inches="tight", pad_inches=0.1
            )
            plt.close(fig)

    # --8<-- [start:model]
    from torch import nn

    from inferno import bnn, loss_fns

    model = bnn.Sequential(
        inferno.models.MLP(
            in_size=2,
            hidden_sizes=[hidden_width] * num_hidden_layers,
            out_size=1,
            activation_layer=nn.SiLU,
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
    loss_fn = loss_fns.BCEWithLogitsLoss()

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

            logits = model(X_batch)

            loss = loss_fn(logits, y_batch)

            loss.backward()
            optimizer.step()
            # --8<-- [end:training]

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

        with torch.no_grad():
            # Model evaluation
            model.eval()

            test_acc_metric = torchmetrics.classification.BinaryAccuracy()

            test_acc_metric.reset()

            for X_batch, y_batch in test_dataloader:

                X_batch = X_batch.to(device=device)
                y_batch = y_batch.to(device=device)

                logits_s = model(X_batch, sample_shape=(16,))

                probs = torch.sigmoid(logits_s).mean(dim=0)
                preds = (probs >= 0.5).float()

                test_acc_metric.update(preds.view(-1), y_batch.view(-1))

            test_acc = test_acc_metric.compute().item()

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
                    "Test Accuracy": test_acc,
                }
            )

    results_df = pd.DataFrame(results_list)

    if plot_learning_curves:
        # Plot learning curves
        with torch.no_grad():
            fig, axs = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(8, 3),
            )

            # Train loss
            sns.lineplot(
                data=results_df,
                x="Epoch",
                y="Train Loss",
                hue="Model",
                ax=axs[0],
                alpha=0.8,
                legend=True,
            )

            # Test accuracy
            sns.lineplot(
                data=results_df,
                x="Epoch",
                y="Test Accuracy",
                hue="Model",
                ax=axs[1],
                alpha=0.8,
                legend=False,
            )

            # Axes
            axs[0].set(yscale="log", title="Train Loss")
            axs[1].set(title="Test Accuracy")

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

            plt.tight_layout()
            plt.savefig(
                pathlib.Path(outdir) / "learning_curves.svg",
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close()

    if plot_decision_boundary:
        # Plot decision boundary
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
        )

        grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)

        grid_t = torch.tensor(grid, dtype=dtype, device=device)

        with torch.no_grad():
            model.eval()
            preds = model(grid_t, sample_shape=(256,))
            probs = torch.sigmoid(preds).mean(dim=0).cpu().numpy().squeeze()

        Z = probs.reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(4, 4))

        cs = ax.contourf(xx, yy, Z, levels=20, alpha=0.8, cmap="RdBu")

        # Training data
        train_scatter = ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap="RdBu",
            edgecolor="k",
            s=30,
            label="Training data",
        )

        # Test data
        test_scatter = ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap="RdBu",
            alpha=0.3,
            s=20,
            label="Test data",
        )

        ax.set_aspect("equal")

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            fancybox=False,
            shadow=False,
            ncol=3,
            frameon=False,
        )

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(
            pathlib.Path(outdir) / "decision_boundary.svg",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()


if __name__ == "__main__":
    fire.Fire(toy_classification)
