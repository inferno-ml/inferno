from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import distributions


class Delta(distributions.Distribution):
    """
    Delta distribution.

    :param param: Parameter of the delta distribution.
    :param atol: Absolute tolerance to consider that a tensor matches the distribution parameter.
    :param rtol: Relative tolerance to consider that a tensor matches the distribution parameter.
    :param batch_shape: The shape over which parameters are batched.
    :param event_shape: The shape of a single sample (without batching).
    """

    arg_constraints: dict = {}

    def __init__(
        self,
        param: torch.Tensor,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        batch_shape: torch.Size | Sequence[int] = None,
        event_shape: torch.Size | Sequence[int] = None,
    ):
        if batch_shape is None:
            batch_shape = torch.Size([])
        if event_shape is None:
            event_shape = torch.Size([])
        self.param = param
        self.atol = atol
        self.rtol = rtol
        if not len(batch_shape) and not len(event_shape):
            batch_shape = param.shape[:-1]
            event_shape = param.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def expand(self, batch_shape: torch.Size, _instance=None) -> Delta:
        if self.batch_shape != tuple(batch_shape):
            return type(self)(
                self.param.expand((*batch_shape, *self.event_shape)),
                atol=self.atol,
                rtol=self.rtol,
            )
        return self

    def _is_equal(self, value: torch.Tensor) -> torch.Tensor:
        param = self.param.expand_as(value)
        is_equal = abs(value - param) < self.atol + self.rtol * abs(param)
        for i in range(-1, -len(self.event_shape) - 1, -1):
            is_equal = is_equal.all(i)
        return is_equal

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        is_equal = self._is_equal(value)
        out = torch.zeros_like(is_equal, dtype=value.dtype)
        out.masked_fill_(is_equal, np.inf)
        out.masked_fill_(~is_equal, -np.inf)
        return out

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return self.param.expand(*sample_shape, *self.param.shape)

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        return self.param.expand(*sample_shape, *self.param.shape)

    @property
    def mode(self) -> torch.Tensor:
        return self.param

    @property
    def mean(self) -> torch.Tensor:
        return self.param
