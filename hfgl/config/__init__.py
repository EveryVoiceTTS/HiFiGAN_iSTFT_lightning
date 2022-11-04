from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Union

from smts.config.preprocessing_config import PreprocessingConfig
from smts.config.shared_types import (
    AdamOptimizer,
    AdamWOptimizer,
    BaseTrainingConfig,
    ConfigModel,
    PartialConfigModel,
    RMSOptimizer,
)
from smts.config.utils import convert_callables
from smts.utils import load_config_from_json_or_yaml_path


class HiFiGANResblock(Enum):
    one = "1"
    two = "2"


class HiFiGANDepthwiseBlocks(ConfigModel):
    """Only currently implemented for the generator"""

    generator: bool


class HiFiGANTrainTypes(Enum):
    original = "original"
    wgan = "wgan"
    wgan_gp = "wgan-gp"


class HiFiGANModelConfig(ConfigModel):
    resblock: HiFiGANResblock
    upsample_rates: List[int]
    upsample_kernel_sizes: List[int]
    upsample_initial_channel: int
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]
    depthwise_separable_convolutions: HiFiGANDepthwiseBlocks
    activation_function: Callable
    istft_layer: bool

    @convert_callables(kwargs_to_convert=["activation_function"])
    def __init__(
        self,
        **data,
    ) -> None:
        """Custom init to process activation function"""
        super().__init__(**data)


class HiFiGANTrainingConfig(BaseTrainingConfig):
    generator_warmup_steps: int
    gan_type: HiFiGANTrainTypes
    optimizer: Union[AdamOptimizer, AdamWOptimizer, RMSOptimizer]
    wgan_clip_value: float
    use_weighted_sampler: bool


class HiFiGANConfig(PartialConfigModel):
    model: HiFiGANModelConfig
    training: HiFiGANTrainingConfig
    preprocessing: PreprocessingConfig

    @staticmethod
    def load_config_from_path(path: Path) -> dict:
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return HiFiGANConfig(**config)


CONFIGS: Dict[str, HiFiGANConfig] = {
    "base": HiFiGANConfig.load_config_from_path(Path(__file__).parent / "base.yaml"),
}
