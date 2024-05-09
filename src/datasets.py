from typing import Any

import kornia.augmentation as K
import torch
import torchgeo.datamodules
import torchgeo.datasets
from torchgeo.transforms import AugmentationSequential


class QuakeSetRegression(torchgeo.datasets.QuakeSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = [d for d in self.data if d["label"] == 1]


class QuakeSetRegressionDataModule(torchgeo.datamodules.QuakeSetDataModule):
    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)

    def __init__(self, *args: Any, image_size: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0)),
            data_keys=["image"],
        )
        self.val_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize((image_size, image_size)),
            data_keys=["image"],
        )
        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize((image_size, image_size)),
            data_keys=["image"],
        )


class QuakeSetDataModule(torchgeo.datamodules.QuakeSetDataModule):
    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["image"],
        )
        self.val_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image"]
        )
        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image"]
        )
