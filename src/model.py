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


class ClassificationRegressionModule(lightning.LightningModule):
    def __init__(
        self,
        classes: list[int],
        image_size: int,
        channels: int,
        backbone: str,
        pretrained: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = len(self.hparams.classes)

    def configure_model(self):
        self.encoder = timm.create_model(
            model_name=self.hparams.backbone,
            pretrained=self.hparams.pretrained,
            img_size=self.hparams.image_size,
        )
        self.cls_head = nn.Linear(
            in_features=self.encoder.num_features, out_features=self.num_classes
        )
        self.reg_head = nn.Linear(in_features=self.encoder.num_features, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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
            {"OverallAccuracy": MeanAbsoluteError(), "OverallF1Score": MeanSquaredError()},
            prefix="train_",
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

    def compute_loss(self, y_pred_cls, y_pred_reg, y_true_cls, y_true_reg):
        loss_cls = F.cross_entropy(y_pred_cls, y_true_cls)
        loss_reg = F.mse_loss(y_pred_reg, y_true_reg)
        return loss_cls + loss_reg

    def training_step(self, batch, batch_idx):
        image, y_true_cls, y_true_reg = (batch["image"], batch["label"], batch["magnitude"])
        y_pred_cls, y_pred_reg = self(image)
        loss = self.regression_loss(y_pred_cls, y_pred_reg, y_true_cls, y_true_reg)
        self.log("train_loss", loss)
        self.train_metrics_cls(y_pred_cls, y_true_cls)
        self.train_metrics_reg(y_pred_reg, y_true_reg)
        return loss

    def validation_step(self, batch, batch_idx):
        image, y_true_cls, y_true_reg = (batch["image"], batch["label"], batch["magnitude"])
        y_pred_cls, y_pred_reg = self(image)
        loss = self.regression_loss(y_pred_cls, y_pred_reg, y_true_cls, y_true_reg)
        self.log("val_loss", loss, on)
        self.train_metrics_cls(y_pred_cls, y_true_cls)
        self.train_metrics_reg(y_pred_reg, y_true_reg)

    def test_step(self, batch, batch_idx):
        sample, label, target = (batch["image"], batch["label"], batch["magnitude"])
        y_pred_cls, y_pred_r = self(sample)
        self.regr_metric(y_r, mag)
        self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        sample = batch["image"]
        y_pred, y_r = self(sample)
        return y_pred, y_r
