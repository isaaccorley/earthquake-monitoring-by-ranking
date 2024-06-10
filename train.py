import argparse

import comet_ml
import hydra
import lightning
import src  # noqa: F401
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger, MLFlowLogger, TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf


@hydra.main(config_path="configs", config_name="baseline", version_base=None)
def main(config):
    lightning.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    datamodule = instantiate(config.datamodule)
    model = instantiate(config.module)

    if config.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir=config.logdir, name=config.experiment_name)
    elif config.logger == "wandb":
        logger = WandbLogger(
            save_dir=config.logdir, name=config.experiment_name, log_modal="all", offline=True
        )
    elif config.logger == "comet":
        logger = CometLogger(
            project_name="earthquake-detection",
            workspace="darthreca",
            experiment_name=config.module.backbone,
            save_dir=config.logdir,
        )
    else:
        logger = MLFlowLogger(
            save_dir=config.logdir, experiment_name=config.experiment_name, log_model="all"
        )

    logger.log_hyperparams(dict(config))

    checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, devices=[config.device]
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
