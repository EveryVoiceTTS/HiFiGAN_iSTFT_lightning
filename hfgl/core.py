from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .config import HiFiGANConfig
    from .model import HiFiGAN, HiFiGANGenerator


def load_config(
    config_file: Path,
    config_args: list[str] = [],
) -> "HiFiGANConfig":
    """Load HiFiGAN configuration from config_file, possibly overriding some parameters"""
    from everyvoice.utils import spinner

    with spinner():
        from everyvoice.base_cli.helpers import load_config_base_command

        from .config import HiFiGANConfig

    config = load_config_base_command(
        model_config=HiFiGANConfig,
        config_file=config_file,
        config_args=config_args,
    )
    assert isinstance(config, HiFiGANConfig)
    return config


def load_model(
    model_path: Path, device: "torch.device"
) -> "HiFiGAN | HiFiGANGenerator":
    """Load a HiFiGAN vocoder checkpoint on device, ready for synthesis.

    Returns a HiFiGANGenerator if the checkpoint only has the generator, or a full
    HiFiGAN (generator and discriminators) otherwise. Either way, the generator
    is in eval mode, without weight norm, and its config is model.config.
    model.global_step_at_load is the training step the checkpoint was saved at.
    """
    from everyvoice.base_cli.checkpoint import load_checkpoint

    from .utils import load_hifigan_from_checkpoint

    checkpoint = load_checkpoint(model_path, minimal=False, device=device)
    model, _config = load_hifigan_from_checkpoint(checkpoint, device)
    return model


PREPROCESS_CATEGORIES = ["audio", "spec"]


def preprocess(
    config: "HiFiGANConfig",
    steps: list[str],
    cpus: int,
    overwrite: bool,
    debug: bool,
):
    """Preprocess audio and text data for HiFiGAN training."""
    from everyvoice.base_cli.helpers import preprocess_base_command

    for step in steps:
        if step not in PREPROCESS_CATEGORIES:
            raise ValueError(f"Unknown step '{step}'")

    preprocessor, _ = preprocess_base_command(
        config=config, steps=steps, cpus=cpus, overwrite=overwrite, debug=debug
    )


def train(
    config: "HiFiGANConfig",
    accelerator: str,
    devices: str,
    nodes: int,
    strategy: str,
):
    """Train a HiFiGAN model"""
    from everyvoice.utils import spinner

    with spinner():
        from everyvoice.base_cli.helpers import train_base_command

        from .dataset import HiFiGANDataModule
        from .model import HiFiGAN

    train_base_command(
        config=config,
        model=HiFiGAN,
        data_module=HiFiGANDataModule,
        monitor="validation/mel_spec_error",
        # We can't do this automatically with Lightning, so we do it manually in model.py
        gradient_clip_val=None,
        accelerator=accelerator,
        devices=devices,
        nodes=nodes,
        strategy=strategy,
    )
