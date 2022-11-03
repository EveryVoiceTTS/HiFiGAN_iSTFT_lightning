import math
import random
from pathlib import Path

import torch
from smts.config.base_config import VocoderConfig
from smts.dataloader import BaseDataModule
from torch.utils.data import Dataset, random_split


class SpecDataset(Dataset):
    def __init__(
        self, audio_files, config: VocoderConfig, use_segments=False, finetune=False
    ):
        self.config = config
        self.sep = config.preprocessing.value_separator
        self.use_segments = use_segments
        self.audio_files = audio_files
        self.preprocessed_dir = Path(self.config.preprocessing.save_dir)
        self.finetune = finetune
        random.seed(self.config.training.seed)
        self.segment_size = self.config.preprocessing.audio.vocoder_segment_size
        self.output_sampling_rate = self.config.preprocessing.audio.output_sampling_rate
        self.input_sampling_rate = self.config.preprocessing.audio.input_sampling_rate
        self.sampling_rate_change = (
            self.output_sampling_rate // self.input_sampling_rate
        )
        self.input_hop_size = self.config.preprocessing.audio.fft_hop_frames
        self.output_hop_size = (
            self.config.preprocessing.audio.fft_hop_frames * self.sampling_rate_change
        )

    def __getitem__(self, index):
        """
        x = mel spectrogram from potentially downsampled audio or from acoustic feature prediction
        y = waveform from potentially upsampled audio
        y_mel = mel spectrogram calculated from y
        """
        item = self.audio_files[index]
        speaker = "default" if "speaker" not in item else item["speaker"]
        language = "default" if "language" not in item else item["language"]
        y = torch.load(
            self.preprocessed_dir
            / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"audio-{self.output_sampling_rate}.npy",
                ]
            )
        ).squeeze()  # [samples] should be output sample rate, squeeze to get rid of channels just in case
        y_mel = torch.load(
            self.preprocessed_dir
            / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"spec-{self.output_sampling_rate}-{self.config.preprocessing.audio.spec_type}.npy",
                ]
            )
        )  # [mel_bins, frames]
        if self.finetune:
            # If finetuning, use the synthesized spectral features
            x = torch.load(
                self.preprocessed_dir
                / self.sep.join(
                    [item["basename"], speaker, language, "spec-synthesized.npy"]
                )
            )
        else:
            x = torch.load(
                self.preprocessed_dir
                / self.sep.join(
                    [
                        item["basename"],
                        speaker,
                        language,
                        f"spec-{self.input_sampling_rate}-{self.config.preprocessing.audio.spec_type}.npy",
                    ]
                )
            )  # [mel_bins, frames]
        frames_per_seg = math.ceil(
            self.segment_size / self.output_hop_size
        )  # segment size is relative to output_sampling_rate, so we use the output_hop_size, but frames_per_seg is in frequency domain, so invariant to x and y_mel
        # other implementations just resample y and take the mel spectrogram of that, but this solution allows for segmenting predicted mel spectrograms from the acoustic feature prediction network too
        if self.use_segments:
            # randomly select a segment, if the segment is too short, pad it with zeros
            if y.size(0) >= self.segment_size:
                max_spec_start = x.size(1) - frames_per_seg - 1
                spec_start = random.randint(0, max_spec_start)
                x = x[:, spec_start : spec_start + frames_per_seg]
                y_mel = y_mel[:, spec_start : spec_start + frames_per_seg]
                y = y[
                    spec_start
                    * self.output_hop_size : (spec_start + frames_per_seg)
                    * self.output_hop_size,
                ]
            else:
                x = torch.nn.functional.pad(
                    x, (0, frames_per_seg - x.size(1)), "constant"
                )
                y_mel = torch.nn.functional.pad(
                    y_mel,
                    (0, frames_per_seg - y_mel.size(1)),
                    "constant",
                )
                y = torch.nn.functional.pad(
                    y, (0, self.segment_size - y.size(0)), "constant"
                )
        return (x, y, self.audio_files[index]["basename"], y_mel)

    def __len__(self):
        return len(self.audio_files)

    def get_labels(self):
        return [x["label"] for x in self.audio_files]


class HiFiGANDataModule(BaseDataModule):
    def __init__(self, config: VocoderConfig):
        super().__init__(config=config)
        self.use_weighted_sampler = config.training.use_weighted_sampler
        self.batch_size = config.training.batch_size
        self.train_split = self.config.training.train_split

    def load_dataset(self):
        self.dataset = self.config.training.filelist_loader(
            self.config.training.filelist
        )

    def prepare_data(self):
        self.load_dataset()
        train_split = int(len(self.dataset) * self.train_split)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_split, len(self.dataset) - train_split]
        )
        self.train_dataset = SpecDataset(
            self.train_dataset, self.config, use_segments=True
        )
        self.val_dataset = SpecDataset(
            self.val_dataset, self.config, use_segments=False
        )
        # save it to disk
        torch.save(self.train_dataset, self.train_path)
        torch.save(self.val_dataset, self.val_path)


class HiFiGANFineTuneDataModule(BaseDataModule):
    def __init__(self, config: VocoderConfig):
        super().__init__(config=config)
        self.use_weighted_sampler = config.training.use_weighted_sampler
        self.batch_size = config.training.batch_size
        self.train_split = self.config.training.train_split

    def load_dataset(self):
        self.dataset = SpecDataset(config=self.config, finetune=True)
