import argparse

import lightning
import src  # noqa: F401
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf


def main(args: argparse.Namespace):
    lightning.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    config = OmegaConf.load(args.config)
    datamodule = instantiate(config.datamodule)
    model = instantiate(config.module)

    if args.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir=args.logdir, name=config.experiment_name)
    elif args.logger == "wandb":
        logger = WandbLogger(
            save_dir=args.logdir, name=config.experiment_name, log_modal="all", offline=True
        )
    else:
        logger = MLFlowLogger(
            save_dir=args.logdir, experiment_name=config.experiment_name, log_model="all"
        )

    logger.log_hyperparams(dict(config))

    checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1, save_last=True, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger, devices=[args.device])
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, help="Path to config file")
    args.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "mlflow", "wandb"]
    )
    args.add_argument("--logdir", type=str, default="./logs", help="Location to save logs")
    args.add_argument("--device", type=int, default=0, help="GPU to use")
    args = args.parse_args()
    main(args)
