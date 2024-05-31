import math
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from everyvoice.config.preprocessing_config import PreprocessingConfig
from everyvoice.config.shared_types import (
    AdamOptimizer,
    AdamWOptimizer,
    BaseModelWithContact,
    BaseTrainingConfig,
    ConfigModel,
    RMSOptimizer,
    init_context,
)
from everyvoice.config.utils import PossiblySerializedCallable, load_partials
from everyvoice.utils import (
    load_config_from_json_or_yaml_path,
    original_hifigan_leaky_relu,
)
from pydantic import (
    Field,
    FilePath,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)


# NOTE: We need to derive from both str and Enum if we want `HiFiGANResblock.one == "one"` to be True.
#    Otherwise, `HiFiGANResblock.one == "one"` will be false.
#    [Python Enum Comparisons](https://docs.python.org/3/howto/enum.html#comparisons)
#    Comparisons against non-enumeration values will always compare not equal.
#    In [1]: from enum import Enum
#    In [2]: class A(Enum):
#       ...:     a = "a"
#       ...:     b = "b"
#       ...:
#    In [3]: A.a == "a"
#    Out[3]: False
#    In [4]: class S(str, Enum):
#       ...:     a = "a"
#       ...:     b = "b"
#       ...:
#    In [5]: S.a == "a"
#    Out[5]: True
# NOTE: The reason behind using an enum is that we actually want to compare to
#    an enum "value" and not to its string representation like so:
#    In [5]: a = A.a
#    In [6]: a is A.a
#    Out[6]: True
class HiFiGANResblock(str, Enum):
    one = "1"
    two = "2"


class HiFiGANTrainTypes(str, Enum):
    original = "original"
    wgan = "wgan"
    # wgan_gp = "wgan-gp"


class HiFiGANModelConfig(ConfigModel):
    resblock: HiFiGANResblock = Field(
        HiFiGANResblock.one,
        description="Which resblock to use. See Kong et. al. 2020: https://arxiv.org/abs/2010.05646",
    )
    upsample_rates: list[int] = Field(
        [8, 8, 2, 2],
        description="The stride of each convolutional layer in the upsampling module.",
    )
    upsample_kernel_sizes: list[int] = Field(
        [16, 16, 4, 4],
        description="The kernel size of each convolutional layer in the upsampling module.",
    )
    upsample_initial_channel: int = Field(
        512,
        description="The number of dimensions to project the Mel inputs to before being passed to the resblock.",
    )
    resblock_kernel_sizes: list[int] = Field(
        [3, 7, 11],
        description="The kernel size of each convolutional layer in the resblock.",
    )
    resblock_dilation_sizes: list[list[int]] = Field(
        [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        description="The dilations of each convolution in each layer of the resblock.",
    )
    activation_function: PossiblySerializedCallable = Field(
        original_hifigan_leaky_relu, description="The activation function to use."
    )
    istft_layer: bool = Field(
        False,
        description="Whether to predict phase and magnitude values and use an inverse Short-Time Fourier Transform instead of predicting a waveform directly. See Kaneko et. al. 2022: https://arxiv.org/abs/2203.02395",
    )
    msd_layers: int = Field(
        3, description="The number of layers to use in the Multi-Scale Discriminator."
    )
    mpd_layers: list[int] = Field(
        [2, 3, 5, 7, 11],
        description="The size of each layer in the Multi-Period Discriminator.",
    )

    @field_serializer("resblock")
    def convert_enum(self, resblock: HiFiGANResblock):
        return resblock.value

    @field_validator("resblock", mode="after")
    @classmethod
    def convert_to_HiFiGANResblock(
        cls,
        v: str | HiFiGANResblock,
        _info: ValidationInfo,
    ) -> HiFiGANResblock:
        return HiFiGANResblock(v)


class HiFiGANTrainingConfig(BaseTrainingConfig):
    generator_warmup_steps: int = Field(
        0,
        description="The number of steps to run through before activating the discriminators.",
    )
    gan_type: HiFiGANTrainTypes = Field(
        HiFiGANTrainTypes.original,
        description="The type of GAN to use. Can be set to either 'original' for a vanilla GAN, or 'wgan' for a Wasserstein GAN that clips gradients.",
    )
    optimizer: AdamOptimizer | AdamWOptimizer | RMSOptimizer = Field(
        default_factory=AdamWOptimizer,
        description="Configuration settings for the optimizer.",
    )
    wgan_clip_value: float = Field(
        0.01, description="The gradient clip value when gan_type='wgan'."
    )
    use_weighted_sampler: bool = Field(
        False,
        description="Whether to use a sampler which oversamples from the minority language or speaker class for balanced training.",
    )
    finetune: bool = Field(
        False,
        description="Whether to read spectrograms from 'preprocessed/synthesized_spec' instead of 'preprocessed/spec'. This is used when finetuning a pretrained spec-to-wav (vocoder) model using the outputs of a trained text-to-spec (feature prediction network) model.",
    )

    @field_serializer("gan_type")
    def convert_enum(self, gan_type: HiFiGANTrainTypes):
        return gan_type.value

    @field_validator("gan_type", mode="after")
    @classmethod
    def convert_to_HiFiGANTrainTypes(
        cls,
        v: str | HiFiGANTrainTypes,
        _info: ValidationInfo,
    ) -> HiFiGANTrainTypes:
        return HiFiGANTrainTypes(v)


class HiFiGANConfig(BaseModelWithContact):
    model: HiFiGANModelConfig = Field(
        default_factory=HiFiGANModelConfig,
        description="The model configuration settings.",
    )
    path_to_model_config_file: Optional[FilePath] = Field(
        None, description="The path of a model configuration file."
    )
    training: HiFiGANTrainingConfig = Field(
        default_factory=HiFiGANTrainingConfig,
        description="The training configuration hyperparameters.",
    )
    path_to_training_config_file: Optional[FilePath] = Field(
        None, description="The path of a training configuration file."
    )
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="The preprocessing configuration, including information about audio settings.",
    )
    path_to_preprocessing_config_file: Optional[FilePath] = Field(
        None, description="The path of a preprocessing configuration file."
    )

    @model_validator(mode="before")  # type: ignore
    def load_partials(self: dict[Any, Any], info: ValidationInfo):
        config_path = (
            info.context.get("config_path", None) if info.context is not None else None
        )
        return load_partials(
            self,
            ("model", "training", "preprocessing"),
            config_path=config_path,
        )

    @staticmethod
    def load_config_from_path(path: Path) -> "HiFiGANConfig":
        """Load a config from a path"""
        config = load_config_from_json_or_yaml_path(path)
        with init_context({"config_path": path}):
            config = HiFiGANConfig(**config)
        return config

    @model_validator(mode="after")
    def check_upsample_rate_consistency(self) -> "HiFiGANConfig":
        # helper variables
        preprocessing_config: PreprocessingConfig = self.preprocessing
        model_config: HiFiGANModelConfig = self.model
        sampling_rate = preprocessing_config.audio.input_sampling_rate
        upsampled_sampling_rate = preprocessing_config.audio.output_sampling_rate
        upsample_rate = upsampled_sampling_rate // sampling_rate
        upsampled_hop_size = upsample_rate * preprocessing_config.audio.fft_hop_size
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
            upsampled_hop_size //= 4  # istft upsamples the rest
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

        return self
