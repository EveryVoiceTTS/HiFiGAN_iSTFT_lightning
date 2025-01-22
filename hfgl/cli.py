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
    vocoder_ckpt = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=True
    )
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
        help="The path to a torch file containing Mel band-oriented spectral features [K (Mel bands), T (frames)]",
        shell_complete=complete_path,
    ),
    generator_path: Path = typer.Option(
        ...,
        "--model",
        "-m",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to a trained EveryVoice spec-to-wav model (i.e., a vocoder)",
        shell_complete=complete_path,
    ),
    time_oriented: bool = typer.Option(
        False,
        help="By default, EveryVoice assumes your spectrograms are of the shape [K (Mel bands), T (frames)]. If instead your spectrograms are of shape [T (frames), K (Mel bands)] then please add this flag to transpose the dimensions.",
    ),
):
    """Given some Mel spectrograms and a trained model, generate some audio. i.e. perform *copy synthesis*."""
    import sys

    with spinner():
        import torch
        import torchaudio
        from pydantic import ValidationError

        from .utils import load_hifigan_from_checkpoint, synthesize_data

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(generator_path, map_location=device, weights_only=True)
    # TODO figure out if we can convert our prepared data format to use weights only
    data = torch.load(data_path, map_location=device, weights_only=False)
    if time_oriented:
        data = data.transpose(0, 1)
    data_size = data.size()
    config_n_mels = checkpoint["hyper_parameters"]["config"]["preprocessing"]["audio"][
        "n_mels"
    ]
    if config_n_mels not in data_size:
        raise ValueError(
            f"Your model expects a spectrogram of dimensions [K (Mel bands), T (frames)] where K == {config_n_mels} but you provided a tensor of size {data_size}"
        )
    if data_size[0] != config_n_mels:
        raise ValueError(
            f"We expected the first dimension of your Mel spectrogram to correspond with the number of Mel bands declared by your model ({config_n_mels}). Instead, we found you model has the dimensions {data_size}. If your spectrogram is time-oriented, please re-run this command with the '--time-oriented' flag."
        )
    try:
        vocoder_model, vocoder_config = load_hifigan_from_checkpoint(checkpoint, device)
    except (TypeError, ValidationError) as e:
        logger.error(f"Unable to load {generator_path}: {e}")
        sys.exit(1)
    wav, sr = synthesize_data(data, vocoder_model, vocoder_config)
    logger.info(f"Writing file {data_path}.wav")
    torchaudio.save(
        f"{data_path}.wav", wav, sr, format="wav", encoding="PCM_S", bits_per_sample=16
    )


if __name__ == "__main__":
    app()
