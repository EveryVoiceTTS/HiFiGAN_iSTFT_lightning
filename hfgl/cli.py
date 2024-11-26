from enum import Enum
from pathlib import Path

import typer
from everyvoice.base_cli.interfaces import (
    complete_path,
    preprocess_base_command_interface,
    train_base_command_interface,
)
from everyvoice.utils import spinner
from loguru import logger
from merge_args import merge_args

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
    help="A PyTorch Lightning implementation of the HiFiGAN and iSTFT-Net vocoders, i.e., spec-to-wav models.",
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
    """Preprocess your data"""
    with spinner():
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
    """Train your spec-to-wav model"""
    with spinner():
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


HFG_EXPORT_SHORT_HELP = (
    "Export and optimize a spec-to-wav model checkpoint for inference"
)
HFG_EXPORT_LONG_HELP = """
    Export your spec-to-wav model.

    # Important!

    This will reduce the size of your checkpoint but it means that the exported checkpoint cannot be resumed for training, it can only be used for inference/synthesis.

    For example:

    **everyvoice export spec-to-wav <path_to_ckpt> <output_path>**
    """


@app.command(
    short_help=HFG_EXPORT_SHORT_HELP,
    help=HFG_EXPORT_LONG_HELP,
)
def export(
    model_path: Path = typer.Argument(
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to a trained EveryVoice spec-to-wav model",
        shell_complete=complete_path,
    ),
    output_path: Path = typer.Option(
        "exported.ckpt",
        "--output",
        "-o",
        exists=False,
        dir_okay=False,
        file_okay=True,
        help="The path to a trained EveryVoice spec-to-wav model",
        shell_complete=complete_path,
    ),
):
    import os

    with spinner():
        import torch

        from .model import HiFiGAN
        from .utils import sizeof_fmt

    orig_size = sizeof_fmt(os.path.getsize(model_path))
    vocoder_ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    HiFiGAN.convert_ckpt_to_generator(vocoder_ckpt)
    torch.save(vocoder_ckpt, output_path)
    new_size = sizeof_fmt(os.path.getsize(output_path))
    logger.info(
        f"Checkpoint saved at '{output_path}'. Reduced size from {orig_size} to {new_size}. This checkpoint will only be usable for inference/synthesis, and not for training."
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
        shell_complete=complete_path,
    ),
    generator_path: Path = typer.Option(
        ...,
        "--model",
        "-m",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to a trained EveryVoice spec-to-wav model",
        shell_complete=complete_path,
    ),
):
    """Given some Mel spectrograms and a trained model, generate some audio. i.e. perform *copy synthesis*"""
    import sys

    with spinner():
        import torch
        from pydantic import ValidationError
        from scipy.io.wavfile import write

        from .utils import load_hifigan_from_checkpoint, synthesize_data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(generator_path, map_location=device)
    data = torch.load(data_path, map_location=device)
    try:
        vocoder_model, vocoder_config = load_hifigan_from_checkpoint(checkpoint, device)
    except (TypeError, ValidationError) as e:
        logger.error(f"Unable to load {generator_path}: {e}")
        sys.exit(1)
    wav, sr = synthesize_data(data, vocoder_model, vocoder_config)
    logger.info(f"Writing file {data_path}.wav")
    write(f"{data_path}.wav", sr, wav)


if __name__ == "__main__":
    app()
