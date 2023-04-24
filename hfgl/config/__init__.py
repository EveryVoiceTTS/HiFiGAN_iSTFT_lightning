import math
from enum import Enum
from pathlib import Path
from typing import Callable, List, Union

from everyvoice.config.preprocessing_config import PreprocessingConfig
from everyvoice.config.shared_types import (
    AdamOptimizer,
    AdamWOptimizer,
    BaseTrainingConfig,
    ConfigModel,
    PartialConfigModel,
    RMSOptimizer,
)
from everyvoice.config.utils import string_to_callable
from everyvoice.utils import load_config_from_json_or_yaml_path, return_configs_from_dir
from pydantic import Field, root_validator, validator


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
    upsample_rates: List[int] = [8, 8]
    upsample_kernel_sizes: List[int] = [16, 16]
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: List[int] = [3, 7, 11]
    resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    depthwise_separable_convolutions: HiFiGANDepthwiseBlocks = Field(
        default_factory=HiFiGANDepthwiseBlocks
    )
    activation_function: Callable = "everyvoice.utils.original_hifigan_leaky_relu"  # type: ignore
    istft_layer: bool = True
    msd_layers: int = 3
    mpd_layers: List[int] = [2, 3, 5, 7, 11]

    @validator("activation_function", pre=True, always=True)
    def convert_callable_activation_function(cls, v, values):
        func = string_to_callable(v)
        values["activation_function"] = func
        return func


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
        sampling_rate = preprocessing_config.audio.input_sampling_rate
        upsampled_sampling_rate = preprocessing_config.audio.output_sampling_rate
        upsample_rate = upsampled_sampling_rate // sampling_rate
        upsampled_hop_size = upsample_rate * preprocessing_config.audio.fft_hop_frames
        upsample_rate_product = math.prod(model_config.upsample_rates)
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
