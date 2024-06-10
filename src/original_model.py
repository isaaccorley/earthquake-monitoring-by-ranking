import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, MeanAbsoluteError, R2Score
from transformers import AutoConfig, AutoModelForImageClassification

from .losses import RegRankLoss


class EarthQuakeModel(LightningModule):
    def __init__(
        self,
        classes: list[int],
        image_size: int,
        channels: int,
        backbone: str,
        pretrained: bool = True,
        lr: float = 1e-4,
        task: str = "regression",
        use_ranking: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        num_classes = 2 if self.hparams["task"] == "classification" else 1

        if "timm" in self.hparams["backbone"]:
            self.model = timm.create_model(
                self.hparams["backbone"],
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=self.hparams["channels"],
                # img_size=image_size,
            )
        else:
            config = AutoConfig.from_pretrained(self.hparams["backbone"])
            config.num_channels = self.hparams["channels"]
            config.num_labels = num_classes
            self.model = AutoModelForImageClassification.from_config(config)

        self.accuracy = Accuracy("multiclass", num_classes=2)
        self.regr_metric = MeanAbsoluteError()
        self.r2_score = R2Score()

        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss() if not self.hparams["use_ranking"] else RegRankLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        if hasattr(x, "logits"):
            x = x.logits
        if self.hparams["task"] == "regression":
            x = torch.clamp(x, 0, 10)
        return x.squeeze()

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

    def training_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"], batch["magnitude"])
        y_r = self(sample)

        loss = 0.0
        if self.hparams["task"] == "classification":
            loss = self.classification_loss(y_r, label)
        elif self.hparams["task"] == "regression":
            loss = self.regression_loss(y_r, mag)
            if isinstance(loss, tuple):
                loss = loss[0]

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"], batch["magnitude"])
        y_r = self(sample)

        loss = 0.0
        if self.hparams["task"] == "regression":
            loss = self.regression_loss(y_r, mag)
            if isinstance(loss, tuple):
                loss = loss[0]

            self.accuracy((y_r >= 1).to(torch.int), label)
            self.log("val_acc", self.accuracy)
            self.regr_metric(y_r, mag)
            self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)
            self.r2_score(y_r, mag)
            self.log("val_r2_score", self.r2_score)
        elif self.hparams["task"] == "classification":
            loss = self.classification_loss(y_r, label)

            self.accuracy(y_r, label)
            self.log("val_acc", self.accuracy)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        sample, label, mag = (batch["image"], batch["label"], batch["magnitude"])

        y_r = self(sample)

        if self.hparams["task"] == "regression":
            self.accuracy((y_r >= 1).to(torch.int), label)
            self.log("val_acc", self.accuracy)
            self.regr_metric(y_r, mag)
            self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)
        elif self.hparams["task"] == "classification":
            self.accuracy(y_r, label)
            self.log("test_acc", self.accuracy)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        sample = batch["image"]
        y_r = self(sample)
        return y_r
