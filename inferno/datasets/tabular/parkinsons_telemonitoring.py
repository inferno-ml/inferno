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


class ParkinsonsTelemonitoring(RegressionDataset):
    """Parkinsons telemonitoring (5,875 × 21).

    This UCI dataset is composed of a range of biomedical voice measurements from 42 people
    with early-stage Parkinson's disease recruited to a six-month trial of a
    telemonitoring device for remote symptom progression monitoring. The recordings were
    automatically captured in the patient's homes. The original study used a range of
    linear and nonlinear regression methods to predict the clinician's Parkinson's
    disease symptom score on the UPDRS scale.

    Source: https://archive.ics.uci.edu/ml/datasets/parkinsons+telemonitoring

    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/"
    md5 = "eba8e7531ac24fbe8473085a0a48e556"

    def __init__(
        self,
        root: str | Path = None,
        transform: Callable | None = transforms.Lambda(
            lambda x: (
                (
                    x
                    - torch.as_tensor(
                        [
                            2.1494e01,
                            6.4805e01,
                            3.1779e-01,
                            9.2864e01,
                            2.1296e01,
                            6.1538e-03,
                            4.4027e-05,
                            2.9872e-03,
                            3.2769e-03,
                            8.9617e-03,
                            3.4035e-02,
                            3.1096e-01,
                            1.7156e-02,
                            2.0144e-02,
                            2.7481e-02,
                            5.1467e-02,
                            3.2120e-02,
                            2.1679e01,
                            5.4147e-01,
                            6.5324e-01,
                            2.1959e-01,
                        ]
                    )
                )
                / torch.as_tensor(
                    [
                        1.2372e01,
                        8.8215e00,
                        4.6566e-01,
                        5.3446e01,
                        8.1293e00,
                        5.6242e-03,
                        3.5983e-05,
                        3.1238e-03,
                        3.7315e-03,
                        9.3715e-03,
                        2.5835e-02,
                        2.3025e-01,
                        1.3237e-02,
                        1.6664e-02,
                        1.9986e-02,
                        3.9711e-02,
                        5.9692e-02,
                        4.2911e00,
                        1.0099e-01,
                        7.0902e-02,
                        9.1498e-02,
                    ]
                )
            ),
        ),
        target_transform: Callable | None = transforms.Lambda(
            lambda y: (y - 29.0189) / 10.7003
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
            / Path("raw/parkinsons_updrs.data")
        )

    def download(self) -> None:
        if not optional_imports_available:
            raise ImportError(optional_imports_not_available_err_msg)

        # Download data
        df = pd.read_csv(self.URL + "parkinsons_updrs.data")
        df.drop(["motor_UPDRS"], axis=1)

        # Move column to predict
        column_to_move = df.pop("total_UPDRS")
        df.insert(0, "total_UPDRS", column_to_move)

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
