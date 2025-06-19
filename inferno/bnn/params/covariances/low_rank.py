from __future__ import annotations

from .factorized import FactorizedCovariance


class LowRankCovariance(FactorizedCovariance):
    r"""Covariance of a Gaussian parameter with low-rank structure.

    Assumes the covariance is factorized as a product of a matrix
    :math:\mathbf{S} \in \mathbb{R}^{P \times R}` and its transpose.

    ..math::
        \mathbf{\Sigma} = \mathbf{S} \mathbf{S}^\top


    :param rank: Rank of the covariance matrix. If `None`, the rank is set to the total number
        of mean parameters.
    """

    def __init__(
        self,
        rank: int,
    ):
        super().__init__(rank=rank)
