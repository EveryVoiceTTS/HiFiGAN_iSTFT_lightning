from enum import Enum
from pathlib import Path

import typer
from everyvoice.base_cli.interfaces import (
    complete_path,
    preprocess_base_command_interface,
    train_base_command_interface,
)
from loguru import logger
from merge_args import merge_args

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="A PyTorch Lightning implementation of the HiFiGAN and iSTFT-Net vocoders",
)


class PreprocessCategories(str, Enum):
    audio = "audio"
    spec = "spec"


@app.command()
@merge_args(preprocess_base_command_interface)
def preprocess(
    steps: list[PreprocessCategories] = typer.Option(
        [cat.value for cat in PreprocessCategories],
        "-s",
        "--steps",
        help="Which steps of the preprocessor to use. If none are provided, all steps will be performed.",
    ),
    **kwargs,
):
    from everyvoice.base_cli.helpers import preprocess_base_command

    from .config import HiFiGANConfig

    preprocess_base_command(
        model_config=HiFiGANConfig,
        steps=[step.name for step in steps],
        **kwargs,
    )


@app.command()
@merge_args(train_base_command_interface)
def train(**kwargs):
    from everyvoice.base_cli.helpers import train_base_command

    from .config import HiFiGANConfig
    from .dataset import HiFiGANDataModule
    from .model import HiFiGAN

    train_base_command(
        model_config=HiFiGANConfig,
        model=HiFiGAN,
        data_module=HiFiGANDataModule,
        monitor="validation/mel_spec_error",
        # We can't do this automatically with Lightning, so we do it manually in model.py
        gradient_clip_val=None,
        **kwargs,
    )


@app.command()
def synthesize(
    data_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to a torch file containing time-oriented spectral features [T (frames), K (Mel bands)]",
        autocompletion=complete_path,
    ),
    generator_path: Path = typer.Option(
        ...,
        "--model",
        "-m",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to a trained EveryVoice spec-to-wav model",
        autocompletion=complete_path,
    ),
):
    """Given some Mel spectrograms and a trained model, generate some audio. i.e. perform *copy synthesis*"""
    import torch
    from scipy.io.wavfile import write

    from .utils import load_hifigan_from_checkpoint, synthesize_data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(generator_path, map_location=device)
    data = torch.load(data_path, map_location=device)
    vocoder_model, vocoder_config = load_hifigan_from_checkpoint(checkpoint, device)
    wav, sr = synthesize_data(data, vocoder_model, vocoder_config)
    logger.info(f"Writing file {data_path}.wav")
    write(f"{data_path}.wav", sr, wav)


if __name__ == "__main__":
    app()
