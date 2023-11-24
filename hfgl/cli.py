from enum import Enum
from pathlib import Path
from typing import List

import typer
from everyvoice.base_cli.interfaces import (
    preprocess_base_command_interface,
    train_base_command_interface,
)
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import check_dataset_size
from loguru import logger
from merge_args import merge_args

from .config import HiFiGANConfig

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="A PyTorch Lightning implementation of the HiFiGAN and iSTFT-Net vocoders",
)


class PreprocessCategories(str, Enum):
    audio = "audio"
    spec = "spec"


@app.command()
@merge_args(preprocess_base_command_interface)
def preprocess(
    steps: List[PreprocessCategories] = typer.Option(
        [cat.value for cat in PreprocessCategories],
        "-s",
        "--steps",
        help="Which steps of the preprocessor to use. If none are provided, all steps will be performed.",
    ),
    **kwargs,
):
    from everyvoice.base_cli.helpers import preprocess_base_command

    preprocess_base_command(
        model_config=HiFiGANConfig,
        steps=[step.name for step in steps],
        **kwargs,
    )


@app.command()
@merge_args(train_base_command_interface)
def train(**kwargs):
    from everyvoice.base_cli.helpers import train_base_command

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
@merge_args(train_base_command_interface)
def match(checkpoint: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to a pre-trained text-to-spec model to match",
    ),**kwargs):
    from everyvoice.base_cli.helpers import train_base_command
    import torch
    from .dataset import HiFiGANDataModule, SpecDataset
    from .model import HiFiGAN

    class HiFiGANFineTuneDataModuleWithCheckpoint(HiFiGANDataModule):
        def __init__(self, config: VocoderConfig):
            super().__init__(config=config)
            self.use_weighted_sampler = config.training.use_weighted_sampler
            self.batch_size = config.training.batch_size
            self.checkpoint = checkpoint

        def prepare_data(self):
            self.load_dataset()
            train_samples = len(self.train_dataset)
            val_samples = len(self.val_dataset)
            check_dataset_size(self.batch_size, train_samples, "training")
            check_dataset_size(self.batch_size, val_samples, "validation")
            self.train_dataset = SpecDataset(
                self.train_dataset, self.config, use_segments=True, checkpoint=checkpoint, finetune=True
            )
            self.val_dataset = SpecDataset(
                self.val_dataset, self.config, use_segments=False, checkpoint=checkpoint, finetune=True
            )
            # save it to disk
            torch.save(self.train_dataset, self.train_path)
            torch.save(self.val_dataset, self.val_path)

    train_base_command(
        model_config=HiFiGANConfig,
        model=HiFiGAN,
        data_module=HiFiGANFineTuneDataModuleWithCheckpoint,
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
        help="The path to some spectral features",
    ),
    generator_path: Path = typer.Option(
        ...,
        "--model",
        "-m",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="The path to a trained model",
    ),
):
    """Given some Mel spectrograms and a trained model, generate some audio. i.e. perform *copy synthesis*"""
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
