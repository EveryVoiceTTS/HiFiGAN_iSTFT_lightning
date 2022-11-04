import json
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from smts.preprocessor import Preprocessor
from smts.utils import expand_config_string_syntax

from hfgl.config import CONFIGS, HiFiGANConfig
from hfgl.dataset import HiFiGANDataModule
from hfgl.model import HiFiGAN

app = typer.Typer(pretty_exceptions_show_locals=False)

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


class PreprocessCategories(str, Enum):
    audio = "audio"
    mel = "mel"


@app.command()
def preprocess(
    name: CONFIGS_ENUM,
    data: Optional[List[PreprocessCategories]] = typer.Option(None, "-d", "--data"),
    output_path: Optional[Path] = typer.Option(
        "processed_filelist.psv", "-o", "--output"
    ),
    overwrite: bool = typer.Option(False, "-O", "--overwrite"),
):
    config = CONFIGS[name.value]
    preprocessor = Preprocessor(config)
    to_preprocess = {k: k in data for k in PreprocessCategories.__members__.keys()}  # type: ignore
    if not data:
        logger.info(
            f"No specific preprocessing data requested, processing everything (pitch, mel, energy, durations, inputs) from dataset '{name}'"
        )
    else:
        preprocessor.preprocess(
            output_path=output_path,
            process_audio=to_preprocess["audio"],
            process_spec=to_preprocess["mel"],
            process_text=to_preprocess["text"],
            overwrite=overwrite,
        )


@app.command()
def train(
    name: CONFIGS_ENUM,
    accelerator: str = typer.Option("auto"),
    devices: str = typer.Option("auto"),
    strategy: str = typer.Option(None),
    config: List[str] = typer.Option(None),
    config_path: Path = typer.Option(None, exists=True, dir_okay=False, file_okay=True),
):
    original_config = CONFIGS[name.value]
    if config is not None and config:
        for update in config:
            key, value = update.split("=")
            logger.info(f"Updating config '{key}' to value '{value}'")
            original_config = original_config.update_config(
                expand_config_string_syntax(update)
            )
    updated_config: HiFiGANConfig = original_config
    if config_path is not None:
        logger.info(f"Loading and updating config from '{config_path}'")
        with open(config_path, "r") as f:
            config_override = (
                json.load(f) if config_path.suffix == ".json" else yaml.safe_load(f)
            )
        updated_config = updated_config.update_config(config_override)
    tensorboard_logger = TensorBoardLogger(**(updated_config.training.logger.dict()))
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger.info("Starting training for HiFiGAN model.")
    ckpt_callback = ModelCheckpoint(
        monitor="validation/mel_spec_error",
        mode="min",
        save_last=True,
        save_top_k=updated_config.training.save_top_k_ckpts,
        every_n_train_steps=updated_config.training.ckpt_steps,
        every_n_epochs=updated_config.training.ckpt_epochs,
    )
    trainer = Trainer(
        logger=tensorboard_logger,
        accelerator=accelerator,
        devices=devices,
        max_epochs=updated_config.training.max_epochs,
        callbacks=[ckpt_callback, lr_monitor],
        strategy=strategy,
        detect_anomaly=False,  # used for debugging, but triples training time
    )
    vocoder = HiFiGAN(updated_config)
    data = HiFiGANDataModule(updated_config)
    last_ckpt = (
        updated_config.training.finetune_checkpoint
        if updated_config.training.finetune_checkpoint is not None
        and os.path.exists(updated_config.training.finetune_checkpoint)
        else None
    )
    tensorboard_logger.log_hyperparams(updated_config.dict())
    trainer.fit(vocoder, data, ckpt_path=last_ckpt)


if __name__ == "__main__":
    app()
