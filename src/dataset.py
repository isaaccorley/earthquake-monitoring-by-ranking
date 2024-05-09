from typing import Any

import kornia.augmentation as K
import torch
from torchgeo.datamodules import QuakeSetDataModule
from torchgeo.transforms import AugmentationSequential


class CustomQuakeSetDataModule(QuakeSetDataModule):
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
