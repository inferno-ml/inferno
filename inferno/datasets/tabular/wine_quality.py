import io
import os
from pathlib import Path
from typing import Callable, Literal
from zipfile import ZipFile

import torch
from torch import Tensor
import torchvision
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


class WineQuality(RegressionDataset):
    """Wine quality prediction from physicochemical properties (4,898 × 11).

    This UCI dataset contains red and white vinho verde wine samples, from the north of Portugal.
    The goal is to model wine quality based on physicochemical tests.

    Source: https://archive.ics.uci.edu/dataset/186/wine+quality
    """

    URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
    md5 = {
        "red": "7d814a1bda02145efe703f4e1c01847a",
        "white": "b56c9a78a7fcad87a58fc586bf5298bc",
    }

    def __init__(
        self,
        root: str | Path = None,
        transform: Callable | None = transforms.Lambda(
            lambda x: (
                (
                    x
                    - torch.as_tensor(
                        [
                            8.3196,
                            0.5278,
                            0.2710,
                            2.5388,
                            0.0875,
                            15.8749,
                            46.4678,
                            0.9967,
                            3.3111,
                            0.6581,
                            10.4230,
                        ]
                    )
                )
                / torch.as_tensor(
                    [
                        1.7411e00,
                        1.7906e-01,
                        1.9480e-01,
                        1.4099e00,
                        4.7065e-02,
                        1.0460e01,
                        3.2895e01,
                        1.8873e-03,
                        1.5439e-01,
                        1.6951e-01,
                        1.0657e00,
                    ]
                )
            ),
        ),
        target_transform: Callable | None = transforms.Lambda(
            lambda y: (y - 5.6360) / 0.8076
        ),
        download: bool = False,
        wine_type: Literal["red", "white"] = "red",
    ):
        self.wine_type = wine_type
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    @property
    def filepath(self) -> str:
        return (
            self.root
            / Path(self.__class__.__name__)
            / Path("raw/winequality-" + self.wine_type + ".csv")
        )

    def download(self) -> None:
        if not optional_imports_available:
            raise ImportError(optional_imports_not_available_err_msg)

        # Download and unzip archive
        r = requests.get(self.URL)
        files = ZipFile(io.BytesIO(r.content))

        # Read data
        df = pd.read_csv(files.open("winequality-" + self.wine_type + ".csv"), sep=";")

        # Save to file
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.filepath, index=False)

    def _check_integrity(self) -> None:
        """Check the integrity of the dataset."""
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(
                "Dataset not found. Use 'download=True' to download the dataset."
            )
        if self.md5 is None:
            return
        if torchvision.datasets.utils.check_md5(
            self.filepath, self.md5[self.wine_type]
        ):
            return
        else:
            raise RuntimeError(
                "Dataset corrupted. Try to delete and re-download the dataset via 'download=True'."
            )

    def _make_dataset(self) -> tuple[Tensor, Tensor]:
        if not optional_imports_available:
            raise ImportError(optional_imports_not_available_err_msg)

        df = pd.read_csv(self.filepath)
        raw_data = torch.as_tensor(df.values).float()
        inputs = raw_data[:, 0:-1]
        targets = raw_data[:, -1]

        return inputs, targets
