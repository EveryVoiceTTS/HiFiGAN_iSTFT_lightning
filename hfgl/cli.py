from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from everyvoice.base_cli.interfaces import (
    preprocess_base_command_interface,
    train_base_command_interface,
)
from loguru import logger
from merge_args import merge_args

from .config import CONFIGS, HiFiGANConfig

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="A PyTorch Lightning implementation of the HiFiGAN and iSTFT-Net vocoders",
)

_config_keys = {k: k for k in CONFIGS.keys()}

CONFIGS_ENUM = Enum("CONFIGS", _config_keys)  # type: ignore


class PreprocessCategories(str, Enum):
    audio = "audio"
    spec = "spec"


@app.command()
@merge_args(preprocess_base_command_interface)
def preprocess(
    name: CONFIGS_ENUM = typer.Option(None, "--name"),
    steps: Optional[List[PreprocessCategories]] = [
        cat.value for cat in PreprocessCategories
    ],
    **kwargs,
):
    from everyvoice.base_cli.helpers import preprocess_base_command

    preprocess_base_command(
        name=name,
        configs=CONFIGS,
        model_config=HiFiGANConfig,
        steps=steps,
        preprocess_categories=PreprocessCategories,
        **kwargs,
    )


@app.command()
@merge_args(train_base_command_interface)
def train(name: CONFIGS_ENUM = typer.Option(None, "--name", "-n"), **kwargs):
    from everyvoice.base_cli.helpers import train_base_command

    from .dataset import HiFiGANDataModule
    from .model import HiFiGAN

    train_base_command(
        name=name,
        model_config=HiFiGANConfig,
        configs=CONFIGS,
        model=HiFiGAN,
        data_module=HiFiGANDataModule,
        monitor="validation/mel_spec_error",
        **kwargs,
    )


@app.command()
def synthesize(
    data_path: Path = typer.Option(
        None, "--input", "-i", exists=True, dir_okay=False, file_okay=True
    ),
    generator_path: Path = typer.Option(
        None, "--model", "-m", exists=True, dir_okay=False, file_okay=True
    ),
):
    import torch
    from scipy.io.wavfile import write

    from .utils import synthesize_data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(generator_path, map_location=device)
    data = torch.load(data_path, map_location=device)
    wav, sr = synthesize_data(data, checkpoint)
    logger.info(f"Writing file {data_path}.wav")
    write(f"{data_path}.wav", sr, wav)


if __name__ == "__main__":
    app()
