import math
import random
from typing import Tuple

import numpy as np
import torch
from everyvoice.utils.heavy import get_spectral_transform
from loguru import logger

from .config import HiFiGANConfig
from .model import HiFiGAN


def load_hifigan_from_checkpoint(ckpt: dict, device) -> Tuple[HiFiGAN, HiFiGANConfig]:
    config: dict | HiFiGANConfig = ckpt["hyper_parameters"]["config"]
    if isinstance(config, dict):
        config = HiFiGANConfig(**config)
    model = HiFiGAN(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.generator.eval()
    model.generator.remove_weight_norm()
    return model, config


def synthesize_data(
    data: torch.Tensor, model: HiFiGAN, config: HiFiGANConfig
) -> Tuple[np.ndarray, int]:
    """Synthesize a batch of waveforms from spectral features

    Args:
        data (Tensor): data tensor, expects output from feature prediction network to be size (b=batch_size, t=number_of_frames, k=n_mels)
        ckpt (dict): HiFiGANLightning checkpoint, expects checkpoint to have a 'hyper_parameters.config' key and HiFiGANConfig object value as well as a 'state_dict' key with model weight as the value
    Returns:
        Tuple[np.ndarray, int]: a B, T array of the synthesized audio and the sampling rate
    """
    if config.model.istft_layer:
        inverse_spectral_transform = get_spectral_transform(
            "istft",
            model.generator.post_n_fft,
            model.generator.post_n_fft,
            model.generator.post_n_fft // 4,
        ).to(data.device)
        with torch.no_grad():
            mag, phase = model.generator(data.transpose(1, 2))
        # We can remove this once the fix for https://github.com/pytorch/pytorch/issues/119088 is merged
        if mag.device.type == "mps" or phase.device.type == "mps":
            logger.warning(
                "Handling complex numbers is broken on MPS (last checked in torch==2.2.0), so we are falling back to CPU"
            )
            mag = mag.to("cpu")
            phase = phase.to("cpu")
            inverse_spectral_transform.to("cpu")
        wavs = inverse_spectral_transform(mag * torch.exp(phase * 1j)).unsqueeze(-2)
    else:
        with torch.no_grad():
            wavs = model.generator(data.transpose(1, 2))
    # squeeze to remove the channel dimension
    return (
        wavs.squeeze(1).cpu().numpy(),
        config.preprocessing.audio.output_sampling_rate,
    )


def get_all_segments(
    x: torch.Tensor, y: torch.Tensor, y_mel: torch.Tensor, segment_size, output_hop_size
):
    """Randomly select a segment from y (wav), x (spectrogram), and y_mel (spectrogram), if the segment is too short, pad it with zeros

    Args:
        x (torch.Tensor): spectrogram input of vocoder
        y (torch.Tensor): waveform output of vocoder
        y_mel (torch.Tensor): spectrogram of waveform output of vocoder
    """
    # segment size is relative to output_sampling_rate, so we use the output_hop_size, but frames_per_seg is in frequency domain, so invariant to x and y_mel
    # other implementations just resample y and take the mel spectrogram of that, but this solution allows for segmenting predicted mel spectrograms from the acoustic feature prediction network too
    frames_per_seg = math.ceil(segment_size / output_hop_size)
    if y.size(0) > segment_size:
        max_spec_start = x.size(1) - frames_per_seg - 1
        spec_start = random.randint(0, max_spec_start)
        x = x[:, spec_start : spec_start + frames_per_seg]
        y_mel = y_mel[:, spec_start : spec_start + frames_per_seg]
        y = y[
            spec_start
            * output_hop_size : (spec_start + frames_per_seg)
            * output_hop_size,
        ]
    else:
        x = torch.nn.functional.pad(x, (0, frames_per_seg - x.size(1)), "constant")
        y_mel = torch.nn.functional.pad(
            y_mel,
            (0, frames_per_seg - y_mel.size(1)),
            "constant",
        )
        y = torch.nn.functional.pad(y, (0, segment_size - y.size(0)), "constant")
    return x, y, y_mel
