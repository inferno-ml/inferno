import abc
import os
from pathlib import Path
from typing import Callable

from torch import Tensor
from torch.utils.data import TensorDataset
import torchvision


class RegressionDataset(TensorDataset, abc.ABC):
    """Abstract base class for regression datasets.

    :param root: Root directory where the dataset is stored.
    :param transform: Transform to apply to the inputs.
    :param target_transform: Transform to apply to the targets.
    :param download: If true, downloads the dataset from the internet and puts it in the root directory.
    """

    def __init__(
        self,
        root: str | Path = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self._check_integrity()
        inputs, targets = self._make_dataset()

        super().__init__(inputs, targets)

    @property
    @abc.abstractmethod
    def filepath(self) -> Path:
        """Filepath of the (raw) data."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def md5(self) -> str:
        """Checksum of the data (archive)."""
        raise NotImplementedError

    @abc.abstractmethod
    def download(self) -> None:
        raise NotImplementedError

    def _check_integrity(self) -> None:
        """Check the integrity of the dataset."""
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(
                "Dataset not found. Use 'download=True' to download the dataset."
            )
        if self.md5 is None:
            return
        if torchvision.datasets.utils.check_md5(self.filepath, self.md5):
            return
        else:
            raise RuntimeError(
                "Dataset corrupted. Try to delete and re-download the dataset via 'download=True'."
            )

    @abc.abstractmethod
    def _make_dataset(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        input, target = super().__getitem__(index)

        # Transform inputs
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return input, target
