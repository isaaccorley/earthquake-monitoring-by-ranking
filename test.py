from argparse import ArgumentParser

import lightning.pytorch as pl
import src
import torch
from torchgeo.datamodules import QuakeSetDataModule


def main(checkpoint: str, device: int):
    torch.set_float32_matmul_precision("medium")
    model = src.original_model.EarthQuakeModel.load_from_checkpoint(checkpoint)
    dm = src.datasets.QuakeSetRegressionDataModule(
        batch_size=32, num_workers=4, root="./data", image_size=512
    )
    trainer = pl.Trainer(accelerator="auto", devices=[device], precision="32-true")
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    main(args.checkpoint, args.device)
