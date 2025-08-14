from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, VisionDataset


class CIFAR10C(VisionDataset):
    """Corrupted CIFAR10 image classification dataset.

    Contains 10,000 test images for each corruption.
    From [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261).

    :param root: Root directory of the dataset.
    :param transform: Transform to apply to the data.
    :param target_transform: Transform to apply to the targets.
    :param corruptions: List of corruptions to apply to the data.
    :param shift_severity: Severity of the corruption to apply.
        Must be an integer between 1 and 5.
    :param download: If true, downloads the dataset from the internet and puts it in the root directory.
    """

    base_folder = Path("CIFAR10-C/raw")
    sub_folder = Path("CIFAR-10-C")
    tgz_md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    filename = "CIFAR-10-C.tar"

    corruption_data_checksums = {
        "fog": "7b397314b5670f825465fbcd1f6e9ccd",
        "jpeg_compression": "2b9cc4c864e0193bb64db8d7728f8187",
        "zoom_blur": "6ea8e63f1c5cdee1517533840641641b",
        "speckle_noise": "ef00b87611792b00df09c0b0237a1e30",
        "glass_blur": "7361fb4019269e02dbf6925f083e8629",
        "spatter": "8a5a3903a7f8f65b59501a6093b4311e",
        "shot_noise": "3a7239bb118894f013d9bf1984be7f11",
        "defocus_blur": "7d1322666342a0702b1957e92f6254bc",
        "elastic_transform": "9421657c6cd452429cf6ce96cc412b5f",
        "gaussian_blur": "c33370155bc9b055fb4a89113d3c559d",
        "frost": "31f6ab3bce1d9934abfb0cc13656f141",
        "saturate": "1cfae0964219c5102abbb883e538cc56",
        "brightness": "0a81ef75e0b523c3383219c330a85d48",
        "snow": "bb238de8555123da9c282dea23bd6e55",
        "gaussian_noise": "ecaf8b9a2399ffeda7680934c33405fd",
        "motion_blur": "fffa5f852ff7ad299cfe8a7643f090f4",
        "contrast": "3c8262171c51307f916c30a3308235a8",
        "impulse_noise": "2090e01c83519ec51427e65116af6b1a",
        "labels": "c439b113295ed5254878798ffe28fd54",
        "pixelate": "0f14f7e2db14288304e1de10df16832f",
    }

    def __init__(
        self,
        root: Path | str = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        corruptions: list[str] = [
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",
            "frost",
            "gaussian_blur",
            "gaussian_noise",
            "glass_blur",
            "impulse_noise",
            "jpeg_compression",
            "motion_blur",
            "pixelate",
            "saturate",
            "shot_noise",
            "snow",
            "spatter",
            "speckle_noise",
            "zoom_blur",
        ],
        shift_severity: int = 5,
        download: bool = False,
    ):
        super().__init__(
            Path(root) / self.base_folder,
            transform=transform,
            target_transform=target_transform,
        )

        self.corruptions = corruptions
        if shift_severity not in list(range(1, 6)):
            raise ValueError(
                "Corruptions 'shift_severity' must be an integer between 1 and 5."
            )
        self.shift_severity = shift_severity

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it."
            )

        # Construct the corrupted dataset
        self.data = np.concatenate(
            [
                np.load(self.root / self.sub_folder / (corruption + ".npy"))[
                    (self.shift_severity - 1) * 10000 : self.shift_severity * 10000
                ]
                for corruption in self.corruptions
            ],
            axis=0,
        )
        self.targets = np.tile(
            np.load(self.root / self.sub_folder / Path("labels.npy"))[
                (self.shift_severity - 1) * 10000 : self.shift_severity * 10000
            ],
            len(self.corruptions),
        ).astype(np.int64)

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset."""
        for corruption in self.corruptions:
            fpath = self.root / self.sub_folder / (corruption + ".npy")
            if corruption not in self.corruptions:
                raise ValueError(f"Unknown corruption '{corruption}'.")
            if not torchvision.datasets.utils.check_integrity(
                fpath, self.corruption_data_checksums[corruption]
            ):
                return False

        return True

    def download(self) -> None:
        if self._check_integrity():
            # Check if files are already downloaded and not corrupted
            return
        else:
            # Download train and test set
            torchvision.datasets.utils.download_and_extract_archive(
                self.url, self.root, filename=self.filename, md5=self.tgz_md5
            )

    def __getitem__(self, index: int) -> tuple[Any, Any]:

        img, target = self.data[index], self.targets[index]

        # Convert to PIL Image (for consistency with other datasets)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class CIFAR100C(CIFAR10C):
    """Corrupted CIFAR100 image classification dataset.

    From [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261).

    :param root: Root directory of the dataset.
    :param transform: Transform to apply to the data.
    :param target_transform: Transform to apply to the targets.
    :param corruptions: List of corruptions to apply to the data.
    :param shift_severity: Severity of the corruption to apply.
        Must be an integer between 1 and 5.
    :param download: If true, downloads the dataset from the internet and puts it in root directory.
    """

    base_folder = Path("CIFAR100-C/raw")
    sub_folder = Path("CIFAR-100-C")
    tgz_md5 = "11f0ed0f1191edbf9fa23466ae6021d3"
    corruption_data_checksums = {
        "fog": "4efc7ebd5e82b028bdbe13048e3ea564",
        "jpeg_compression": "c851b7f1324e1d2ffddeb76920576d11",
        "zoom_blur": "0204613400c034a81c4830d5df81cb82",
        "speckle_noise": "e3f215b1a0f9fd9fd6f0d1cf94a7ce99",
        "glass_blur": "0bf384f38e5ccbf8dd479d9059b913e1",
        "spatter": "12ccf41d62564d36e1f6a6ada5022728",
        "shot_noise": "b0a1fa6e1e465a747c1b204b1914048a",
        "defocus_blur": "d923e3d9c585a27f0956e2f2ad832564",
        "elastic_transform": "a0792bd6581f6810878be71acedfc65a",
        "gaussian_blur": "5204ba0d557839772ef5a4196a052c3e",
        "frost": "3a39c6823bdfaa0bf8b12fe7004b8117",
        "saturate": "c0697e9fdd646916a61e9c312c77bf6b",
        "brightness": "f22d7195aecd6abb541e27fca230c171",
        "snow": "0237be164583af146b7b144e73b43465",
        "gaussian_noise": "ecc4d366eac432bdf25c024086f5e97d",
        "motion_blur": "732a7e2e54152ff97c742d4c388c5516",
        "contrast": "322bb385f1d05154ee197ca16535f71e",
        "impulse_noise": "3b3c210ddfa0b5cb918ff4537a429fef",
        "labels": "bb4026e9ce52996b95f439544568cdb2",
        "pixelate": "96c00c60f144539e14cffb02ddbd0640",
    }
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    filename = "CIFAR-100-C.tar"
