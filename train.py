import argparse
import time

import lightning
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf


def main(args: argparse.Namespace):
    lightning.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    config = OmegaConf.load(args.config)
    datamodule = instantiate(config.datamodule)
    model = instantiate(config.module)

    experiment_id = time.strftime("%Y%m%d-%H%M%S")

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{config.module.backbone}/{experiment_id}",
        save_top_k=1,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    trainer = instantiate(config.trainer, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config")
    main(args)
