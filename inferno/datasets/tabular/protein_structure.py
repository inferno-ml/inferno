from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torchvision.transforms import v2 as transforms

from .. import RegressionDataset

try:
    import pandas as pd

    optional_imports_available = True
except ImportError:
    optional_imports_available = False
optional_imports_not_available_err_msg = "Pandas not found. Install inferno with dataset dependencies via 'pip install inferno[datasets]'."


class ProteinStructure(RegressionDataset):
    """Physicochemical properties of protein tertiary structure (45,730 × 9).

    This UCI dataset encompasses the physicochemical properties of protein tertiary structure, sourced from CASP 5-9.
    There are 45,730 decoys with 9 attributes and sizes varying from 0 to 21 angstroms.

    Source: https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/"
    md5 = "2cd0971a73f135ceb6aae74fe724a6f5"

    def __init__(
        self,
        root: str | Path = None,
        transform: Callable | None = transforms.Lambda(
            lambda x: (
                (
                    x
                    - torch.as_tensor(
                        [
                            9.8716e03,
                            3.0174e03,
                            3.0239e-01,
                            1.0349e02,
                            1.3683e06,
                            1.4564e02,
                            3.9898e03,
                            6.9975e01,
                            3.4524e01,
                        ]
                    )
                )
                / torch.as_tensor(
                    [
                        4.0581e03,
                        1.4643e03,
                        6.2886e-02,
                        5.5425e01,
                        5.6404e05,
                        6.9999e01,
                        1.9936e03,
                        5.6493e01,
                        5.9798e00,
                    ]
                )
            ),
        ),
        target_transform: Callable | None = transforms.Lambda(
            lambda y: (y - 7.7485) / 6.1183
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
        return self.root / Path(self.__class__.__name__) / Path("raw/CASP.csv")

    def download(self) -> None:
        if not optional_imports_available:
            raise ImportError(optional_imports_not_available_err_msg)

        # Download data
        df = pd.read_csv(self.URL + "CASP.csv")

        # Save to file
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.filepath, index=False)

    def _make_dataset(self) -> tuple[Tensor, Tensor]:
        if not optional_imports_available:
            raise ImportError(optional_imports_not_available_err_msg)

        df = pd.read_csv(self.filepath)
        raw_data = torch.as_tensor(df.values).float()
        inputs = raw_data[:, 1::]
        targets = raw_data[:, 0]

        return inputs, targets
