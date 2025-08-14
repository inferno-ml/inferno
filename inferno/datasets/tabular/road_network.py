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


class RoadNetwork(RegressionDataset):
    """3D Road Network (434,874 × 2).

    This UCI Dataset contains longitude, latitude and altitude values of a road network in
    North Jutland, Denmark (covering a region of 185x135 km2). Elevation values where
    extracted from a publicly available massive Laser Scan Point Cloud for Denmark.
    The regression task is to predict the altitude from longitude and latitude measurements.

    Source: https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland,+Denmark)
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00246/"
    md5 = "989a6f4574e09ee6735d8af2e5885cc1"

    def __init__(
        self,
        root: str | Path = None,
        transform: Callable | None = transforms.Lambda(
            lambda x: (
                (x - torch.as_tensor([9.7318, 57.0838]))
                / torch.as_tensor([0.6273, 0.2895])
            ),
        ),
        target_transform: Callable | None = transforms.Lambda(
            lambda y: (y - 22.1854) / 18.6180
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
        return (
            self.root
            / Path(self.__class__.__name__)
            / Path("raw/3D_spatial_network.txt")
        )

    def download(self) -> None:
        if not optional_imports_available:
            raise ImportError(optional_imports_not_available_err_msg)

        # Download data
        df = pd.read_csv(
            RoadNetwork.URL + "3D_spatial_network.txt",
            header=None,
            names=["OSM_ID", "longitude", "latitude", "altitude"],
        )

        # Save to file
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.filepath, index=False)

    def _make_dataset(self) -> tuple[Tensor, Tensor]:
        if not optional_imports_available:
            raise ImportError(optional_imports_not_available_err_msg)

        df = pd.read_csv(self.filepath)
        raw_data = torch.as_tensor(df.values).float()
        inputs = raw_data[:, 1:-1]
        targets = raw_data[:, -1]

        return inputs, targets
