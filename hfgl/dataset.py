from pathlib import Path

import torch
import torchaudio
from everyvoice.dataloader import BaseDataModule
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import check_dataset_size
from torch.utils.data import Dataset

from .utils import get_all_segments


class SpecDataset(Dataset):
    def __init__(
        self, audio_files, config: VocoderConfig, use_segments=False, finetune=False
    ):
        self.config = config
        self.sep = "--"
        self.use_segments = use_segments
        self.audio_files = audio_files
        self.preprocessed_dir = Path(self.config.preprocessing.save_dir)
        self.finetune = self.config.training.finetune
        self.segment_size = self.config.preprocessing.audio.vocoder_segment_size
        self.output_sampling_rate = self.config.preprocessing.audio.output_sampling_rate
        self.input_sampling_rate = self.config.preprocessing.audio.input_sampling_rate
        self.sampling_rate_change = (
            self.output_sampling_rate // self.input_sampling_rate
        )
        self.input_hop_size = self.config.preprocessing.audio.fft_hop_size
        self.output_hop_size = (
            self.config.preprocessing.audio.fft_hop_size * self.sampling_rate_change
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
        y, _ = torchaudio.load(
            (
                self.config.preprocessing.save_dir
                / "audio"
                / self.sep.join(
                    [
                        item["basename"],
                        speaker,
                        language,
                        f"audio-{self.output_sampling_rate}.wav",
                    ]
                )
            ).__str__()
        )
        y = (
            y.squeeze()
        )  # [samples] should be output sample rate, squeeze to get rid of channels
        y_mel = torch.load(
            self.preprocessed_dir
            / "spec"
            / self.sep.join(
                [
                    item["basename"],
                    speaker,
                    language,
                    f"spec-{self.output_sampling_rate}-{self.config.preprocessing.audio.spec_type}.pt",
                ]
            )
        )  # [mel_bins, frames]
        if self.finetune:
            # If finetuning, use the synthesized spectral features
            x = torch.load(
                self.preprocessed_dir
                / "synthesized_spec"
                / self.sep.join(
                    [
                        item["basename"],
                        speaker,
                        language,
                        f"spec-pred-{self.input_sampling_rate}-{self.config.preprocessing.audio.spec_type}.pt",
                    ]
                )
            ).transpose(0, 1)
        else:
            x = torch.load(
                self.preprocessed_dir
                / "spec"
                / self.sep.join(
                    [
                        item["basename"],
                        speaker,
                        language,
                        f"spec-{self.input_sampling_rate}-{self.config.preprocessing.audio.spec_type}.pt",
                    ]
                )
            )  # [mel_bins, frames]
        if self.use_segments:
            x, y, y_mel = get_all_segments(
                x, y, y_mel, self.segment_size, self.output_hop_size
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

    def load_dataset(self):
        self.train_dataset = self.config.training.filelist_loader(
            self.config.training.training_filelist
        )
        self.val_dataset = self.config.training.filelist_loader(
            self.config.training.validation_filelist
        )

    def prepare_data(self):
        self.load_dataset()
        train_samples = len(self.train_dataset)
        val_samples = len(self.val_dataset)
        check_dataset_size(self.batch_size, train_samples, "training")
        check_dataset_size(self.batch_size, val_samples, "validation")
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

    def load_dataset(self):
        self.dataset = SpecDataset(config=self.config, finetune=True)
