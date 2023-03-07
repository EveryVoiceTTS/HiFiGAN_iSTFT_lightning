import math
import random
from typing import Tuple

import numpy as np
import torch
from everyvoice.utils.heavy import get_spectral_transform

from .model import Generator


def synthesize_data(data: torch.Tensor, generator_ckpt: dict) -> Tuple[np.ndarray, int]:
    """Synthesize a batch of waveforms from spectral features

    Args:
        data_path (Path): path to data tensor. expects output from feature prediction network to be size (b=batch_size, t=number_of_frames, k=n_mels)
        generator_path (Path): path to HiFiGANLightning checkpoint. expects checkpoint to have a 'config' key and HiFiGANConfig object value as well as a 'state_dict' key with model weight as the value
    Returns:
        Tuple[np.ndarray, int]: a 1-D array of the wav file and the sampling rate
    """
    config = generator_ckpt["config"]
    model = Generator(config).to(data.device)
    model.load_state_dict(generator_ckpt["state_dict"])
    model.eval()
    model.remove_weight_norm()
    if config.model.istft_layer:
        inverse_spectral_transform = get_spectral_transform(
            "istft", model.post_n_fft, model.post_n_fft, model.post_n_fft // 4
        ).to(data.device)
        with torch.no_grad():
            mag, phase = model(data.transpose(1, 2))
        wav = inverse_spectral_transform(mag * torch.exp(phase * 1j)).unsqueeze(-2)
    else:
        with torch.no_grad():
            wav = model(data.transpose(1, 2))
    return (
        wav.squeeze().cpu().numpy(),
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
    if y.size(0) >= segment_size:
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
