from .covariances.diagonal import DiagonalCovariance
from .covariances.factorized import FactorizedCovariance
from .covariances.kronecker import KroneckerCovariance
from .covariances.low_rank import LowRankCovariance
from .gaussian_parameter import GaussianParameter
from .parameter import BNNParameter
from .parametrizations import MaximalUpdate, NeuralTangent, Parametrization, Standard

SP = Standard
NTP = NeuralTangent
MUP = MaximalUpdate
