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
    min = torch.tensor([0.0, 0.0, 0.0, 0.0])
    max = torch.tensor([1.0265753e-02, 2.1116707e02, 4.2103156e-01, 1.1088742e00])
    mean = torch.tensor([2.6106297e-03, 1.5721272e01, 8.2029007e-02, 1.2758774e-01])
    std = torch.tensor([5.6622936e-03, 1.7652454e02, 1.1999646e-01, 7.3716629e-01])

    db_min = torch.tensor([-90.0, -90.0, -90.0, 0.0])
    db_max = torch.tensor([-19.886091, 23.246262, -8.204279, 28.966887])
    db_mean = torch.tensor([-44.725605, -3.6441572, -16.686876, 3.5217872])
    db_std = torch.tensor([82.02653, 20.370014, 35.2053, 13.023926])

    def __init__(self, *args: Any, image_size: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.val_aug = AugmentationSequential(
            ToDecibel(),
            # K.Normalize(mean=self.mean, std=self.std),
            # K.Resize((image_size, image_size)),
            data_keys=["image"],
        )
        self.train_aug = AugmentationSequential(
            ToDecibel(),
            # K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            # K.Resize((image_size, image_size)),
            data_keys=["image"],
        )


class QuakeSetDataModule(torchgeo.datamodules.QuakeSetDataModule):
    min = torch.tensor([0.0, 0.0, 0.0, 0.0])
    max = torch.tensor([1.0265753e-02, 2.1116707e02, 4.2103156e-01, 1.1088742e00])
    mean = torch.tensor([2.6106320e-03, 1.5721265e01, 8.2028978e-02, 1.2758777e-01])
    std = torch.tensor([5.6622936e-03, 1.7652454e02, 1.1999646e-01, 7.3716629e-01])

    db_min = torch.tensor([-90.0, -90.0, -90.0, 0.0])
    db_max = torch.tensor([-19.886091, 23.246262, -8.204279, 28.966887])
    db_mean = torch.tensor([-44.725613, -3.6441576, -16.686846, 3.5217931])
    db_std = torch.tensor([82.02653, 20.370014, 35.2053, 13.023926])

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


class ToDecibel(K.IntensityAugmentationBase2D):
    def __init__(self, p: float = 1, keepdim: bool = False) -> None:
        super().__init__(same_on_batch=True, p=p, keepdim=keepdim)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return 10 * torch.log10(input + 1e-9)
