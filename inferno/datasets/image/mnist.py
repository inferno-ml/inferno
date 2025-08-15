from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import MNIST, VisionDataset


class MNISTC(VisionDataset):
    """Corrupted MNIST image classification dataset.

    Contains 10,000 test images for each one of 15 corruptions.
    From [MNIST-C: A Robustness Benchmark for Computer Vision](https://arxiv.org/abs/1906.02337).

    :param root: Root directory of the dataset.
    :param transform: Transform to apply to the data.
    :param target_transform: Transform to apply to the targets.
    :param corruptions: List of corruptions to apply to the data.
    :param download: If true, downloads the dataset from the internet and puts it in the root directory.
    """

    base_folder = Path("MNIST-C/raw")
    sub_folder = Path("mnist_c")
    zip_md5 = "4b34b33045869ee6d424616cd3a65da3"
    url = "https://zenodo.org/record/3239543/files/mnist_c.zip"
    filename = "mnist_c.zip"

    def __init__(
        self,
        root: Path | str = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        corruptions: list[str] = [
            "brightness",
            "canny_edges",
            "dotted_line",
            "fog",
            "glass_blur",
            "impulse_noise",
            "motion_blur",
            "rotate",
            "scale",
            "shear",
            "shot_noise",
            "spatter",
            "stripe",
            "translate",
            "zigzag",
        ],
        download: bool = False,
    ):
        super().__init__(
            Path(root) / self.base_folder,
            transform=transform,
            target_transform=target_transform,
        )

        self.corruptions = corruptions

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it."
            )

        # Construct the corrupted dataset
        self.data = np.concatenate(
            [
                np.load(self.root / self.sub_folder / corruption / "test_images.npy")
                for corruption in self.corruptions
            ],
            axis=0,
        )
        self.targets = np.tile(
            np.load(self.root / self.sub_folder / "identity/test_labels.npy"),
            len(self.corruptions),
        ).astype(np.int64)

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset."""
        return torchvision.datasets.utils.check_integrity(
            self.root / self.filename, self.zip_md5
        )

    def download(self) -> None:
        if self._check_integrity():
            # Check if files are already downloaded and not corrupted
            return
        else:
            # Download train and test set
            torchvision.datasets.utils.download_and_extract_archive(
                self.url, self.root, filename=self.filename, md5=self.zip_md5
            )

    def __getitem__(self, index: int) -> tuple[Any, Any]:

        img, target = self.data[index], self.targets[index]

        # Convert to PIL Image (for consistency with other datasets)
        img = Image.fromarray(img[..., -1], mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
