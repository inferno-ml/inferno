import io
import os
from pathlib import Path
from typing import Callable
from zipfile import ZipFile

import torch
from torch import Tensor
from torchvision.transforms import v2 as transforms

from .. import RegressionDataset

try:
    import pandas as pd
    import requests

    optional_imports_available = True
except ImportError:
    optional_imports_available = False
optional_imports_not_available_err_msg = (
    "Pandas or requests libraries not found. "
    "Install inferno with dataset dependencies via 'pip install inferno[datasets]'."
)


class ConcreteCompressiveStrength(RegressionDataset):
    """Concrete compressive strength (1,030 × 8).

    This UCI dataset contains the ingredients of concrete mixtures and their age. The regression task is to predict the
    concrete's compressive strength.

    Source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
    """

    URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
    md5 = "4aaeecaf0bf2eefccb8a4a6d4cc12785"

    def __init__(
        self,
        root: str | Path = None,
        transform: Callable | None = transforms.Lambda(
            lambda x: (
                (
                    x
                    - torch.as_tensor(
                        [
                            281.1656,
                            73.8955,
                            54.1871,
                            181.5664,
                            6.2031,
                            972.9186,
                            773.5789,
                            45.6621,
                        ]
                    )
                )
                / torch.as_tensor(
                    [
                        104.5071,
                        86.2791,
                        63.9965,
                        21.3556,
                        5.9735,
                        77.7538,
                        80.1754,
                        63.1699,
                    ]
                )
            ),
        ),
        target_transform: Callable | None = transforms.Lambda(
            lambda y: (y - 35.8178) / 16.7057
        ),
        download: bool = False,
    ):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    @property
    def filepath(self) -> str:
        return self.root / Path(self.__class__.__name__) / Path("raw/Concrete_Data.xls")

    def download(self) -> None:
        if not optional_imports_available:
            raise ImportError(optional_imports_not_available_err_msg)

        # Download and unzip archive
        r = requests.get(self.URL)
        files = ZipFile(io.BytesIO(r.content))

        # Read data
        df = pd.read_excel(files.open("Concrete_Data.xls"))

        # Save to file
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.filepath, index=False)

    def _make_dataset(self) -> tuple[Tensor, Tensor]:
        if not optional_imports_available:
            raise ImportError(optional_imports_not_available_err_msg)

        df = pd.read_csv(self.filepath)
        raw_data = torch.as_tensor(df.values).float()
        inputs = raw_data[:, 0:-1]
        targets = raw_data[:, -1]

        return inputs, targets
