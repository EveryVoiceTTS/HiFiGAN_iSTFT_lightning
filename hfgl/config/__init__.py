import math
from enum import Enum
from pathlib import Path
from typing import Callable, List, Union

from pydantic import Field, root_validator
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
from smts.utils import load_config_from_json_or_yaml_path, return_configs_from_dir


class HiFiGANResblock(Enum):
    one = "1"
    two = "2"


class HiFiGANDepthwiseBlocks(ConfigModel):
    """Only currently implemented for the generator"""

    generator: bool = False


class HiFiGANTrainTypes(Enum):
    original = "original"
    wgan = "wgan"
    wgan_gp = "wgan-gp"


class HiFiGANModelConfig(ConfigModel):
    resblock: HiFiGANResblock = HiFiGANResblock.one
    super_resolution_rate: int = 1
    upsample_rates: List[int] = [8, 8]
    upsample_kernel_sizes: List[int] = [16, 16]
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: List[int] = [3, 7, 11]
    resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    depthwise_separable_convolutions: HiFiGANDepthwiseBlocks = Field(
        default_factory=HiFiGANDepthwiseBlocks
    )
    activation_function: Callable = "smts.utils.original_hifigan_leaky_relu"  # type: ignore
    istft_layer: bool = True

    @convert_callables(kwargs_to_convert=["activation_function"])
    def __init__(
        self,
        **data,
    ) -> None:
        """Custom init to process activation function"""
        super().__init__(**data)


class HiFiGANFreezingLayers(ConfigModel):
    all_layers: bool = False
    msd: bool = False
    mpd: bool = False
    generator: bool = False


class HiFiGANTrainingConfig(BaseTrainingConfig):
    generator_warmup_steps: int = 0
    gan_type: HiFiGANTrainTypes = HiFiGANTrainTypes.original
    optimizer: Union[AdamOptimizer, AdamWOptimizer, RMSOptimizer] = Field(
        default_factory=AdamWOptimizer
    )
    wgan_clip_value: float = 0.01
    use_weighted_sampler: bool = False
    freeze_layers: HiFiGANFreezingLayers = Field(default_factory=HiFiGANFreezingLayers)
    finetune: bool = False


class HiFiGANConfig(PartialConfigModel):
    model: HiFiGANModelConfig = Field(default_factory=HiFiGANModelConfig)
    training: HiFiGANTrainingConfig = Field(default_factory=HiFiGANTrainingConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)

    @staticmethod
    def load_config_from_path(path: Path) -> "HiFiGANConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        return HiFiGANConfig(**config)

    @root_validator
    def check_upsample_rate_consistency(cls, values):
        # helper variables
        preprocessing_config: PreprocessingConfig = values["preprocessing"]
        model_config: HiFiGANModelConfig = values["model"]
        super_resolution_rate = model_config.super_resolution_rate
        sampling_rate = preprocessing_config.audio.input_sampling_rate
        upsampled_sampling_rate = preprocessing_config.audio.output_sampling_rate
        upsample_rate = upsampled_sampling_rate // sampling_rate
        upsampled_hop_size = upsample_rate * preprocessing_config.audio.fft_hop_frames
        upsample_rate_product = math.prod(model_config.upsample_rates) * super_resolution_rate

        # check super resolution rate
        if super_resolution_rate == 0:
            raise ValueError('Super resolution rate cannot be 0, please set to 1 instead')
        if (super_resolution_rate > 1 and super_resolution_rate % 2) != 0:
            raise ValueError('Super resolution currently only works for even multiples. Please set to an even number')
        # check that same number of kernels and kernel sizes exist
        if len(model_config.upsample_kernel_sizes) != len(model_config.upsample_rates):
            raise ValueError(
                "Number of upsample kernel sizes must match number of upsample rates"
            )
        # Check that kernel sizes are not less than upsample rates and are evenly divisible
        for kernel_size, upsample_rate in zip(
            model_config.upsample_kernel_sizes, model_config.upsample_rates
        ):
            if kernel_size < upsample_rate:
                raise ValueError(
                    f"Upsample kernel size: {kernel_size} must be greater than upsample rate: {upsample_rate}"
                )
            if kernel_size % upsample_rate != 0:
                raise ValueError(
                    f"Upsample kernel size: {kernel_size} must be evenly divisible by upsample rate: {upsample_rate}"
                )
        # check that upsample rate is even multiple of target sampling rate
        if upsampled_sampling_rate % sampling_rate != 0:
            raise ValueError(
                f"Target sampling rate: {upsampled_sampling_rate} must be an even multiple of input sampling rate: {sampling_rate}"
            )
        # check that the upsampling hop size is equal to product of upsample rates
        if model_config.istft_layer:
            upsampled_hop_size /= 4  # istft upsamples the rest
        # check that upsampled hop size is equal to product of upsampling rates
        if upsampled_hop_size != upsample_rate_product:
            raise ValueError(
                f"Upsampled hop size: {upsampled_hop_size} must be equal to product of upsample rates: {upsample_rate_product}"
            )
        # check that the segment size is divisible by product of upsample rates
        if preprocessing_config.audio.vocoder_segment_size % upsample_rate_product != 0:
            raise ValueError(
                f"Vocoder segment size: {preprocessing_config.audio.vocoder_segment_size} must be divisible by product of upsample rates: {upsample_rate_product}"
            )

        return values


CONFIG_DIR = Path(__file__).parent
CONFIGS = return_configs_from_dir(CONFIG_DIR)
