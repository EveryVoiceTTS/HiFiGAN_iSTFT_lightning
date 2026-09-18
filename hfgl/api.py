import multiprocessing as mp
from pathlib import Path

from . import core
from .config import HiFiGANConfig


def load_config(
    config_file: Path | str,
) -> HiFiGANConfig:
    """Load a HiFiGAN configuration from config_file.

    Your hfgl config file is called "config/everyvoice-spec-to-wav.yaml" if it
    was created using the "everyvoice new-project" wizard.

    If you need to override any values, change them in the returned config object.

    Args:
        config_file (Path|str): HiFiGAN configuration filename
    """
    return core.load_config(config_file=Path(config_file))


def preprocess(
    config: HiFiGANConfig,
    steps: list[str] = core.PREPROCESS_CATEGORIES,
    cpus: int = min(4, mp.cpu_count()),
    overwrite: bool = False,
    debug: bool = False,
) -> None:
    """Preprocess data for spec-to-wav (HiFiGAN) training.

    The datasets to process are described in config.

    Args:
        config (HiFiGANConfig): your HiFiGAN configuration
        steps (list[str]): steps to process, one or more of "audio", "spec"
        cpus (int): how many CPUs to use for preprocessing
        overwrite (bool): if false, existing files will be kept and only new files will be generated;
                   if true, redo all preprocessing, even if files already exist
        debug (bool): enable debugging
    """
    core.preprocess(
        config=config,
        steps=steps,
        cpus=cpus,
        overwrite=overwrite,
        debug=debug,
    )


def train(
    config: HiFiGANConfig,
    accelerator: str = "auto",
    devices: str | int = "auto",
    nodes: int = 1,
    strategy: str = "ddp",
):
    """Train your Spec-to-Wav (HiFiGAN) model

    Args:
        config (HiFiGANConfig): your HiFiGAN configuration
        accelerator (str): PyTorch Lightning Accelerator to use: https://pytorch-lightning.readthedocs.io/en/stable/extensions/accelerator.html
        devices ("auto" | str | int): the number of GPUs to use on each node as a str or int; use "auto" to let pytoch-lightning decide
        nodes (int): the number of nodes to use
        strategy (str): the strategy for data parallelization: https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_intermediate.html"
    """
    core.train(
        config=config,
        accelerator=accelerator,
        devices=str(devices),
        nodes=nodes,
        strategy=strategy,
    )
