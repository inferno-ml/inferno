from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class TemperatureScaler(nn.Module):
    """Temperature scaling.

    Tunes the temperature parameter of the model in the last layer to minimize the negative
    log likelihood of the validation set.

    Based on [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).

    :param loss_fn: The loss function to be used for calibration.
    :param lr: Learning rate for the optimizer.
    :param max_iter: Maximum number of iterations per optimization step.
    :param tolerance_grad: Tolerance for the gradient.
    :param tolerance_change: Tolerance for the change in the loss function / parameters.
    :param history_size: Size of the history for the LBFGS optimizer.
    """

    def __init__(
        self,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        lr: float = 0.1,
        max_iter: int = 100,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
    ) -> None:
        super().__init__()
        if not isinstance(loss_fn, (nn.BCEWithLogitsLoss, nn.CrossEntropyLoss)):
            raise ValueError(
                f"Loss function must be either BCEWithLogitsLoss or CrossEntropyLoss,"
                f" but is {loss_fn.__class__.__name__}."
            )
        self.lr = lr
        self.loss_fn = loss_fn
        self.max_iter = max_iter
        self.history_size = history_size
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change

    def optimize(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """Optimizes the temperature of the model.

        :param model: The model to be calibrated, assumed to return logits.
        :param dataloader: The dataloader for the dataset to calibrate on (typically the validation set).
        """
        # Set the model to evaluation mode to ensure temperature scaling is applied
        model_was_in_training_mode = False
        if model.training:
            model_was_in_training_mode = True
            model.eval()

        # Freeze all parameters except the temperature parameter
        requires_grad_dict = {}
        temperature_parameter = None
        for name, param in model.named_parameters():
            requires_grad_dict[name] = param.requires_grad
            if "temperature" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
                temperature_parameter = param

        if temperature_parameter is None:
            raise ValueError("Model does not have a temperature parameter.")

        # Optimizer
        optimizer = torch.optim.LBFGS(
            params=[temperature_parameter],
            lr=self.lr,
            max_iter=self.max_iter,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            history_size=self.history_size,
            # line_search_fn="strong_wolfe",
        )

        # Define the closure function for the optimizer
        def closure() -> Tensor:
            optimizer.zero_grad()

            # Get all batches from the validation data
            all_logits = []
            all_targets = []
            for inputs, targets in dataloader:
                logits = model(
                    inputs.to(temperature_parameter.device)
                )  # TODO: how should we pick the number of samples?
                all_logits.append(logits)
                all_targets.append(targets)
            all_logits = torch.cat(all_logits).to(temperature_parameter.device)
            all_targets = torch.cat(all_targets).to(temperature_parameter.device)

            # Compute the loss
            loss = self.loss_fn(all_logits, all_targets)

            loss.backward()
            return loss

        # Optimize temperature
        optimizer.step(closure)

        # Set requires_grad to what it was prior to temperature scaling
        for name, param in model.named_parameters():
            param.requires_grad_(requires_grad_dict[name])
        temperature_parameter.requires_grad_(False)

        if model_was_in_training_mode:
            model.train()
