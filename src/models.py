import lightning
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import (
    Accuracy,
    ClasswiseWrapper,
    FBetaScore,
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    Precision,
    Recall,
)

from .losses import RegRankLoss


class ClassificationRegressionModule(lightning.LightningModule):
    def __init__(
        self,
        classes: list[int],
        image_size: int,
        channels: int,
        backbone: str,
        pretrained: bool = True,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = len(self.hparams.classes)
        self.configure_model()
        self.configure_metrics()

    def configure_model(self):
        kwargs = {}
        if "swin" in self.hparams.backbone or "vit" in self.hparams.backbone:
            kwargs["img_size"] = self.hparams.image_size

        self.encoder = timm.create_model(
            model_name=self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            in_chans=self.hparams.channels,
            num_classes=0,
            **kwargs,
        )
        self.cls_head = nn.Linear(
            in_features=self.encoder.num_features, out_features=self.num_classes
        )
        self.reg_head = nn.Linear(in_features=self.encoder.num_features, out_features=1)

    def configure_metrics(self):
        self.train_metrics_cls = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    task="multiclass", num_classes=self.num_classes, average="micro"
                ),
                "OverallF1Score": FBetaScore(
                    task="multiclass", num_classes=self.num_classes, beta=1.0, average="micro"
                ),
                "OverallPrecision": Precision(
                    task="multiclass", num_classes=self.num_classes, average="micro"
                ),
                "OverallRecall": Recall(
                    task="multiclass", num_classes=self.num_classes, average="micro"
                ),
                "AverageAccuracy": Accuracy(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
                "AverageF1Score": FBetaScore(
                    task="multiclass", num_classes=self.num_classes, beta=1.0, average="macro"
                ),
                "AveragePrecision": Precision(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
                "AverageRecall": Recall(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
                "Accuracy": ClasswiseWrapper(
                    Accuracy(task="multiclass", num_classes=self.num_classes, average="none"),
                    labels=self.hparams.classes,
                ),
                "Precision": ClasswiseWrapper(
                    Precision(task="multiclass", num_classes=self.num_classes, average="none"),
                    labels=self.hparams.classes,
                ),
                "Recall": ClasswiseWrapper(
                    Recall(task="multiclass", num_classes=self.num_classes, average="none"),
                    labels=self.hparams.classes,
                ),
                "F1Score": ClasswiseWrapper(
                    FBetaScore(
                        task="multiclass", num_classes=self.num_classes, beta=1.0, average="none"
                    ),
                    labels=self.hparams.classes,
                ),
            },
            prefix="train_",
        )
        self.train_metrics_reg = MetricCollection(
            {"MAE": MeanAbsoluteError(), "MSE": MeanSquaredError()}, prefix="train_"
        )
        self.val_metrics_cls = self.train_metrics_cls.clone(prefix="val_")
        self.test_metrics_cls = self.train_metrics_cls.clone(prefix="test_")
        self.val_metrics_reg = self.train_metrics_reg.clone(prefix="val_")
        self.test_metrics_reg = self.train_metrics_reg.clone(prefix="test_")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.hparams["lr"],
            pct_start=0.1,
            cycle_momentum=False,
            div_factor=1e9,
            final_div_factor=1e4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        c = self.cls_head(x)
        r = self.reg_head(x).squeeze(dim=1)
        return c, r

    def compute_loss(self, y_c, y_r, y_true_c, y_true_r):
        loss_c = F.cross_entropy(y_c, y_true_c)
        loss_r = F.mse_loss(y_r, y_true_r)
        return loss_c + loss_r

    def training_step(self, batch, batch_idx):
        image, y_true_c, y_true_r = batch["image"], batch["label"], batch["magnitude"]
        y_c, y_r = self(image)
        loss = self.regression_loss(y_c, y_r, y_true_c, y_true_r)
        self.log("train_loss", loss)
        self.train_metrics_cls(y_c, y_true_c)
        self.train_metrics_reg(y_r, y_true_r)
        self.log_dict({f"{k}": v for k, v in self.train_metrics_cls.compute().items()})
        self.log_dict({f"{k}": v for k, v in self.train_metrics_reg.compute().items()})
        return loss

    def validation_step(self, batch, batch_idx):
        image, y_true_c, y_true_r = batch["image"], batch["label"], batch["magnitude"]
        y_c, y_r = self(image)
        loss = self.regression_loss(y_c, y_r, y_true_c, y_true_r)
        self.log("val_loss", loss, on_epoch=True)
        self.val_metrics_cls(y_c, y_true_c)
        self.val_metrics_reg(y_r, y_true_r)
        self.log_dict({f"{k}": v for k, v in self.val_metrics_cls.compute().items()})
        self.log_dict({f"{k}": v for k, v in self.val_metrics_reg.compute().items()})

    def test_step(self, batch, batch_idx):
        image, y_true_c, y_true_r = batch["image"], batch["label"], batch["magnitude"]
        y_c, y_r = self(image)
        self.test_metrics_cls(y_c, y_true_c)
        self.test_metrics_reg(y_r, y_true_r)
        self.log_dict({f"{k}": v for k, v in self.test_metrics_cls.compute().items()})
        self.log_dict({f"{k}": v for k, v in self.test_metrics_reg.compute().items()})

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        image = batch["image"]
        y_c, y_r = self(image)
        return y_c, y_r


class ClassificationRankingRegressionModule(ClassificationRegressionModule):
    def __init__(self, *args, margin=0.02, **kwargs):
        super().__init__()
        self.loss_fn = RegRankLoss(margin=margin)

    def compute_loss(self, y_c, y_r, y_true_c, y_true_r):
        loss_c = F.cross_entropy(y_c, y_true_c)
        loss_r, _, _ = self.loss_fn(y_r, y_true_r)
        return loss_c + loss_r


class RegressionModule(lightning.LightningModule):
    def __init__(
        self,
        image_size: int,
        channels: int,
        backbone: str,
        pretrained: bool = True,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.configure_model()
        self.configure_metrics()

    def configure_model(self):
        kwargs = {}
        if "swin" in self.hparams.backbone or "vit" in self.hparams.backbone:
            kwargs["img_size"] = self.hparams.image_size

        self.encoder = timm.create_model(
            model_name=self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            in_chans=self.hparams.channels,
            num_classes=0,
            **kwargs,
        )
        self.reg_head = nn.Linear(in_features=self.encoder.num_features, out_features=1)

    def configure_metrics(self):
        self.train_metrics = MetricCollection(
            {"MAE": MeanAbsoluteError(), "MSE": MeanSquaredError()}, prefix="train_"
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.hparams["lr"],
            pct_start=0.1,
            cycle_momentum=False,
            div_factor=1e9,
            final_div_factor=1e4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        r = self.reg_head(x).squeeze(dim=1)
        return r

    def compute_loss(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def training_step(self, batch, batch_idx):
        image, y_true = batch["image"], batch["magnitude"]
        y_pred = self(image)
        loss = self.compute_loss(y_pred, y_true)
        self.log("train_loss", loss)
        self.train_metrics(y_pred, y_true)
        self.log_dict({f"{k}": v for k, v in self.train_metrics.compute().items()})
        return loss

    def validation_step(self, batch, batch_idx):
        image, y_true = batch["image"], batch["magnitude"]
        y_pred = self(image)
        loss = self.compute_loss(y_pred, y_true)
        self.log("val_loss", loss, on_epoch=True)
        self.val_metrics(y_pred, y_true)
        self.log_dict({f"{k}": v for k, v in self.val_metrics.compute().items()})

    def test_step(self, batch, batch_idx):
        image, y_true = batch["image"], batch["magnitude"]
        y_pred = self(image)
        self.test_metrics(y_pred, y_true)
        self.log_dict({f"{k}": v for k, v in self.test_metrics.compute().items()})

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        image = batch["image"]
        y_pred = self(image)
        return y_pred


class RankingRegressionModule(RegressionModule):
    def __init__(self, *args, margin: float = 0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = RegRankLoss(margin=margin)

    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)[0]
