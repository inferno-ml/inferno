from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import torchvision
from torchvision.datasets import VisionDataset


class TinyImageNet(VisionDataset):
    """TinyImageNet image classification dataset.

    The training dataset contains 100,000 images of 200 classes (500 for each class) downsized to 64x64 color images.
    The test set has 10,000 images (50 for each class).

    :param root: Root directory of the dataset.
    :param train: If True, creates dataset from training data, otherwise from test data.
    :param transform: Transform to apply to the data.
    :param target_transform: Transform to apply to the targets.
    :param download: If true, downloads the dataset from the internet and puts it in the root directory.
    """

    base_folder = Path("TinyImageNet/raw")
    tgz_md5 = "90528d7ca1a48142e341f4ef8d21d0de"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        root = Path(root)

        super().__init__(
            root / self.base_folder,
            transform=transform,
            target_transform=target_transform,
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                f"Dataset '{self.__class__.__name__}' not found or corrupted. You can use download=True to download it."
            )

        self._image_root = root / self.base_folder / Path("tiny-imagenet-200")
        self._split = "train" if train else "val"

        # Construct the dataset
        with open(self._image_root / "wnids.txt") as r:
            classes = list(map(lambda s: s.strip(), r.readlines()))

        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.data = []
        data_dir = root / self.base_folder / Path("tiny-imagenet-200") / self._split
        if self._split == "train":
            for fname in data_dir.iterdir():
                if fname.is_dir():
                    image_folder = fname / Path("images")
                    for img_path in image_folder.iterdir():
                        self.data.append((img_path, self.class_to_idx[fname.stem]))
        else:
            imgs_path = os.path.join(data_dir, "images")
            imgs_annotations = os.path.join(data_dir, "val_annotations.txt")

            with open(imgs_annotations) as r:
                data_info = map(lambda s: s.split("\t"), r.readlines())

            cls_map = {line_data[0]: line_data[1] for line_data in data_info}

            for imgname in sorted(os.listdir(imgs_path)):
                path = os.path.join(imgs_path, imgname)
                item = (path, self.class_to_idx[cls_map[imgname]])
                self.data.append(item)

        self.targets = [item[1] for item in self.data]

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset."""
        return torchvision.datasets.utils.check_integrity(
            Path(self.root) / Path(self.filename), self.tgz_md5
        )

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
        img_path, target = self.data[index]
        image = torchvision.datasets.folder.default_loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)


class TinyImageNetC(VisionDataset):
    """Corrupted TinyImageNet image classification dataset.

    Contains 10,000 64x64 color test images for each corruption (200 classes, 50 images per class).
    From [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261).

    :param root: Root directory of the dataset.
    :param transform: Transform to apply to the data.
    :param target_transform: Transform to apply to the targets.
    :param corruptions: List of corruptions to apply to the data.
    :param shift_severity: Severity of the corruption to apply.
        Must be an integer between 1 and 5.
    :param download: If true, downloads the dataset from the internet and puts it in the root directory.
    """

    base_folder = Path("TinyImageNet-C/raw")
    tgz_md5 = [
        "f9c9a9dbdc11469f0b850190f7ad8be1",
        "0db0588d243cf403ef93449ec52b70eb",
    ]
    url = "https://zenodo.org/record/8206060/files/"
    filename = ["Tiny-ImageNet-C.tar", "Tiny-ImageNet-C-extra.tar"]

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
                f"Dataset '{self.__class__.__name__}' not found or corrupted. You can use download=True to download it."
            )

        # Construct the corrupted dataset
        self.data = []
        self.targets = []
        tiny_image_net = TinyImageNet(root, download=download)
        for corruption in self.corruptions:
            data_dir = (
                root
                / self.base_folder
                / Path(corruption)
                / Path(str(self.shift_severity))
            )
            for filename in data_dir.iterdir():
                if filename.is_dir():
                    for img_path in filename.iterdir():
                        self.data.append(img_path)
                        self.targets.append(tiny_image_net.class_to_idx[filename.stem])

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset."""
        for filename, md5 in list(zip(self.filename, self.tgz_md5, strict=True)):
            fpath = self.root / filename
            if not torchvision.datasets.utils.check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        """Download the dataset."""
        if self._check_integrity():
            return
        for filename, md5 in list(zip(self.filename, self.tgz_md5, strict=True)):
            torchvision.datasets.utils.download_and_extract_archive(
                self.url + filename,
                self.root,
                filename=filename,
                md5=md5,
            )

        # Remove unnecessary subfolder
        for filename in (self.root / Path("Tiny-ImageNet-C")).iterdir():
            path = Path(filename).absolute()
            parent_dir = path.parents[1]
            path.rename(parent_dir / path.name)
        (self.root / Path("Tiny-ImageNet-C")).rmdir()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        img_path, target = self.data[index], self.targets[index]
        image = torchvision.datasets.folder.default_loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.data)
