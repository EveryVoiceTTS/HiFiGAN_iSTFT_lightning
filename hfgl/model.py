import itertools
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from everyvoice.model.vocoder.config import VocoderConfig
from everyvoice.utils import plot_spectrogram
from everyvoice.utils.heavy import (
    dynamic_range_compression_torch,
    get_spectral_transform,
)
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from .config import HiFiGANResblock, HiFiGANTrainTypes


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
    elif classname.find("Sequential") != -1:
        for layer in m:
            layer.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(
        self,
        config: VocoderConfig,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
    ):
        super(ResBlock1, self).__init__()
        self.config = config
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=x,
                        padding=get_padding(kernel_size, x),
                    )
                )
                for x in dilation
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in dilation
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = self.config.model.activation_function(x)
            xt = c1(xt)
            xt = self.config.model.activation_function(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            if layer.__class__.__name__ == "Sequential":
                for sub_layer in layer:
                    remove_parametrizations(sub_layer, "weight")
            else:
                remove_parametrizations(layer, "weight")
        for layer in self.convs2:
            if layer.__class__.__name__ == "Sequential":
                for sub_layer in layer:
                    remove_parametrizations(sub_layer, "weight")
            else:
                remove_parametrizations(layer, "weight")


class ResBlock2(torch.nn.Module):
    def __init__(
        self,
        config: VocoderConfig,
        channels,
        kernel_size=3,
        dilation=(1, 3),
    ):
        super(ResBlock2, self).__init__()
        self.config = config
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = self.config.model.activation_function(x)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            if layer.__class__.__name__ == "Sequential":
                for sub_layer in layer:
                    remove_parametrizations(sub_layer, "weight")
            else:
                remove_parametrizations(layer, "weight")


class Generator(torch.nn.Module):
    def __init__(self, config: VocoderConfig):
        super(Generator, self).__init__()
        self.config = config
        self.audio_config = config.preprocessing.audio
        self.sampling_rate_change = (
            self.audio_config.output_sampling_rate
            // self.audio_config.input_sampling_rate
        )
        self.model_vocoder_config = config.model
        self.num_kernels = len(self.model_vocoder_config.resblock_kernel_sizes)
        self.num_upsamples = len(self.model_vocoder_config.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(
                self.audio_config.n_mels,
                self.model_vocoder_config.upsample_initial_channel,
                7,
                1,
                padding=3,
            )
        )

        resblock = (
            ResBlock1
            if self.model_vocoder_config.resblock is HiFiGANResblock.one
            else ResBlock2
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(
                self.model_vocoder_config.upsample_rates,
                self.model_vocoder_config.upsample_kernel_sizes,
            )
        ):
            # TODO: add sensible upsampling layer
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        self.model_vocoder_config.upsample_initial_channel
                        // (2**i),  # in
                        self.model_vocoder_config.upsample_initial_channel
                        // (2 ** (i + 1)),  # out
                        k,  # kernel
                        u,  # stride
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.model_vocoder_config.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(
                self.model_vocoder_config.resblock_kernel_sizes,
                self.model_vocoder_config.resblock_dilation_sizes,
            ):
                self.resblocks.append(resblock(self.config, ch, k, d))
        if self.config.model.istft_layer:
            self.post_n_fft = (
                self.audio_config.n_fft * self.sampling_rate_change
            ) // math.prod(self.config.model.upsample_rates)
            self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
            conv_post_out_channels = self.post_n_fft + 2
        else:
            self.post_n_fft = conv_post_out_channels = 1

        self.conv_post = weight_norm(
            Conv1d(ch, conv_post_out_channels, 7, 1, padding=3)
        )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.config.model.activation_function(x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        # NOTE: Changed from user provided activation to fixed leaky_relu to mimic our reference code in jik876.
        x = F.leaky_relu(x)
        if self.config.model.istft_layer:
            x = self.reflection_pad(x)
            x = self.conv_post(x)
            spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
            phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
            return spec, phase
        else:
            x = self.conv_post(x)
            x = torch.tanh(x)
            return x

    def remove_weight_norm(self):
        for layer in self.ups:
            if layer.__class__.__name__ == "Sequential":
                for sub_layer in layer:
                    remove_parametrizations(sub_layer, "weight")
            else:
                remove_parametrizations(layer, "weight")
        for layer in self.resblocks:
            layer.remove_weight_norm()
        if self.conv_pre.__class__.__name__ == "Sequential":
            for sub_layer in self.conv_pre:
                remove_parametrizations(sub_layer, "weight")
        else:
            remove_parametrizations(self.conv_pre, "weight")
        if self.conv_post.__class__.__name__ == "Sequential":
            for sub_layer in self.conv_post:
                remove_parametrizations(sub_layer, "weight")
        else:
            remove_parametrizations(self.conv_post, "weight")


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self, period, config, kernel_size=5, stride=3, use_spectral_norm=False
    ):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.config = config
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = self.config.model.activation_function(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, config):
        super(MultiPeriodDiscriminator, self).__init__()
        mpd_layers = [DiscriminatorP(n, config) for n in config.model.mpd_layers]
        self.discriminators = nn.ModuleList(mpd_layers)

    def forward_interpolates(self, interp):
        y_ds = []
        f_maps = []
        for d in self.discriminators:
            y_d, f_map = d(interp)
            y_ds.append(y_d)
            f_maps.append(f_map)
        return y_ds, f_maps

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, config, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        self.config = config
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = self.config.model.activation_function(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, config: VocoderConfig):
        super(MultiScaleDiscriminator, self).__init__()
        msd_layers = [
            DiscriminatorS(config, use_spectral_norm=i == 0)
            for i in range(config.model.msd_layers)
        ]
        self.discriminators = nn.ModuleList(msd_layers)
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2) for _ in range(config.model.msd_layers - 1)]
        )

    def forward_interpolates(self, interp):
        y_ds = []
        f_maps = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                interp = self.meanpools[i - 1](interp)
            y_d, f_map = d(interp)
            y_ds.append(y_d)
            f_maps.append(f_map)
        return y_ds, f_maps

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class HiFiGAN(pl.LightningModule):
    def __init__(self, config: dict | VocoderConfig):
        super().__init__()
        # Required by PyTorch Lightning for our manual optimization
        self.automatic_optimization = False
        # Because we serialize the configurations when saving checkpoints,
        # sometimes what is passed is actually just a dict.
        if not isinstance(config, VocoderConfig):
            config = VocoderConfig(**config)
        self.config = config
        self.mpd = MultiPeriodDiscriminator(config)
        self.msd = MultiScaleDiscriminator(config)
        self.generator = Generator(config)
        self.save_hyperparameters()  # TODO: ignore=['specific keys'] - I should ignore some unnecessary/problem values
        self.update_config_settings()
        if self.config.model.istft_layer:
            self.inverse_spectral_transform = get_spectral_transform(
                "istft",
                self.generator.post_n_fft,
                self.generator.post_n_fft,
                self.generator.post_n_fft // 4,
            )
        # TODO: figure out multiple nodes/gpus: https://pytorch-lightning.readthedocs.io/en/1.4.0/advanced/multi_gpu.html

    def update_config_settings(self):
        # batch_size is declared explicitly so that auto_scale_batch_size works:
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html
        if self.config.training.finetune:
            for disc in self.mpd.discriminators:
                for layer in disc.convs[:3]:
                    for p in layer.parameters():
                        p.requires_grad = False
            for disc in self.msd.discriminators:
                for layer in disc.convs[:4]:
                    for p in layer.parameters():
                        p.requires_grad = False
        self.batch_size = self.config.training.batch_size
        self.audio_config = self.config.preprocessing.audio
        self.sampling_rate_change = (
            self.audio_config.output_sampling_rate
            // self.audio_config.input_sampling_rate
        )
        self.use_wgan = self.config.training.gan_type is HiFiGANTrainTypes.wgan
        # We don't have to set the fft size and hop/window lengths as hyperparameters here, because we can just multiply by the upsampling rate
        self.spectral_transform = get_spectral_transform(
            self.audio_config.spec_type,
            self.audio_config.n_fft * self.sampling_rate_change,
            self.audio_config.fft_window_size * self.sampling_rate_change,
            self.audio_config.fft_hop_size * self.sampling_rate_change,
            f_min=self.audio_config.f_min,
            f_max=self.audio_config.f_max,
            sample_rate=self.audio_config.output_sampling_rate,
            n_mels=self.audio_config.n_mels,
        )

    def forward(self, x):
        return self.generator(x)

    def on_load_checkpoint(self, checkpoint):
        """Deserialize the checkpoint hyperparameters.
        Note, this shouldn't fail on different versions of pydantic anymore,
        but it will fail on breaking changes to the config. We should catch those exceptions
        and handle them appropriately."""
        self.config = VocoderConfig(**checkpoint["hyper_parameters"]["config"])

    def on_save_checkpoint(self, checkpoint):
        """Serialize the checkpoint hyperparameters"""
        checkpoint["hyper_parameters"]["config"] = self.config.model_checkpoint_dump()

    def configure_optimizers(self):
        generator_params = self.generator.parameters()
        if self.config.training.finetune:
            msd_params = filter(lambda p: p.requires_grad, self.msd.parameters())
            mpd_params = filter(lambda p: p.requires_grad, self.mpd.parameters())
        else:
            msd_params = self.msd.parameters()
            mpd_params = self.mpd.parameters()
        if self.config.training.optimizer.name == "adamw":
            optim_g = torch.optim.AdamW(
                generator_params,
                self.config.training.optimizer.learning_rate,
                betas=[
                    self.config.training.optimizer.betas[0],
                    self.config.training.optimizer.betas[1],
                ],
            )
            optim_d = torch.optim.AdamW(
                itertools.chain(msd_params, mpd_params),
                self.config.training.optimizer.learning_rate,
                betas=[
                    self.config.training.optimizer.betas[0],
                    self.config.training.optimizer.betas[1],
                ],
            )
        elif self.config.training.optimizer.name == "rms":
            optim_g = torch.optim.RMSprop(
                generator_params,
                lr=self.config.training.optimizer.learning_rate,
            )
            optim_d = torch.optim.RMSprop(
                itertools.chain(msd_params, mpd_params),
                lr=self.config.training.optimizer.learning_rate,
            )
        if self.use_wgan:
            return (
                {"optimizer": optim_g, "frequency": 1},
                {"optimizer": optim_d, "frequency": 5},
            )
        else:
            scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                optim_g,
                gamma=0.999,  # TODO: parametrize this
            )
            scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999)
            return [optim_g, optim_d], [scheduler_g, scheduler_d]

    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2

    def discriminator_loss(
        self,
        disc_real_outputs,
        disc_generated_outputs,
    ):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

    def generator_loss(self, disc_outputs):
        g_loss = 0
        gen_losses = []
        for dg in disc_outputs:
            loss = torch.mean((1 - dg) ** 2)
            gen_losses.append(loss)
            g_loss += loss

        return (g_loss, gen_losses)

    def training_step(self, batch, batch_idx):
        x, y, _, y_mel = batch
        y = y.unsqueeze(1)
        # x.size() & y_mel.size() = [batch_size, n_mels=80, n_frames=32]
        # y.size() = [batch_size, segment_size=8192]
        optim_g, optim_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()
        # generate waveform
        if self.config.model.istft_layer:
            mag, phase = self(x)
            generated_wav = self.inverse_spectral_transform(
                mag * torch.exp(phase * 1j)
            ).unsqueeze(-2)
        else:
            generated_wav = self(x)

        # create mel
        generated_mel_spec = dynamic_range_compression_torch(
            self.spectral_transform(generated_wav).squeeze(1)[:, :, 1:]
        )
        # train discriminators
        if self.global_step >= self.config.training.generator_warmup_steps:
            optim_d.zero_grad()
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, generated_wav.detach())
            loss_disc_f, _, _ = self.discriminator_loss(y_df_hat_r, y_df_hat_g)
            self.log("training/disc/mpd_loss", loss_disc_f, prog_bar=False)
            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, generated_wav.detach())
            loss_disc_s, _, _ = self.discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            self.log("training/disc/msd_loss", loss_disc_s, prog_bar=False)
            # WGAN
            if self.use_wgan:
                for p in self.msd.parameters():
                    p.data.clamp_(
                        -self.config.training.wgan_clip_value,
                        self.config.training.wgan_clip_value,
                    )
                for p in self.mpd.parameters():
                    p.data.clamp_(
                        -self.config.training.wgan_clip_value,
                        self.config.training.wgan_clip_value,
                    )
            # calculate loss
            disc_loss_total = loss_disc_s + loss_disc_f
            # manual optimization because Pytorch Lightning 2.0+ doesn't handle automatic optimization for multiple optimizers
            # use .backward for now, but maybe switch to self.manual_backward() in the future: https://github.com/Lightning-AI/lightning/issues/18740
            # self.manual_backward(disc_loss_total)
            disc_loss_total.backward()
            # clip gradients
            self.clip_gradients(
                optim_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            optim_d.step()
            # step in the scheduler every epoch
            if self.trainer.is_last_batch:
                scheduler_d.step()
            # log discriminator loss
            self.log("training/disc/d_loss_total", disc_loss_total, prog_bar=False)

        # train generator
        optim_g.zero_grad()
        # calculate loss
        loss_mel = F.l1_loss(y_mel, generated_mel_spec) * 45
        if self.global_step >= self.config.training.generator_warmup_steps:
            _, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, generated_wav)
            _, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, generated_wav)
            loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g)
            # loss_gen_f = -torch.mean(y_df_hat_g)
            # loss_gen_s = -torch.mean(y_ds_hat_g)
            loss_gen_f, _ = self.generator_loss(y_df_hat_g)
            loss_gen_s, _ = self.generator_loss(y_ds_hat_g)
            self.log("training/gen/loss_fmap_f", loss_fm_f, prog_bar=False)
            self.log("training/gen/loss_fmap_s", loss_fm_s, prog_bar=False)
            self.log("training/gen/loss_gen_f", loss_gen_f, prog_bar=False)
            self.log("training/gen/loss_gen_s", loss_gen_s, prog_bar=False)
            gen_loss_total = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
        else:
            gen_loss_total = loss_mel
        # manual optimization because Pytorch Lightning 2.0+ doesn't handle automatic optimization for multiple optimizers
        # use .backward for now, but maybe switch to self.manual_backward() in the future: https://github.com/Lightning-AI/lightning/issues/18740
        # self.manual_backward(gen_loss_total)
        gen_loss_total.backward()
        # clip gradients
        self.clip_gradients(
            optim_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        optim_g.step()
        # step in the scheduler every epoch
        if self.trainer.is_last_batch:
            scheduler_g.step()
        # log generator loss
        self.log("training/gen/gen_loss_total", gen_loss_total, prog_bar=False)
        self.log("training/gen/mel_spec_error", loss_mel / 45, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x, y, bn, y_mel = batch
        current_step = (
            self.global_step // 2
        )  # because self.global_step counts gen/disc steps
        # generate waveform
        if self.config.model.istft_layer:
            mag, phase = self(x)
            generated_wav = self.inverse_spectral_transform(
                mag * torch.exp(phase * 1j)
            ).unsqueeze(-2)
        else:
            generated_wav = self(x)
        # create mel
        generated_mel_spec = dynamic_range_compression_torch(
            self.spectral_transform(generated_wav).squeeze(1)[:, :, 1:]
        )
        # Since we are not using fixed-size segments, sometimes the prediction is off by one frame when doing super resolution/upsampling
        val_err_tot = F.l1_loss(y_mel, generated_mel_spec[:, :, : y_mel.size(2)]).item()
        # # Below is taken from HiFiGAN
        if self.global_step == 0:
            # Log ground truth audio and spec
            self.logger.experiment.add_audio(
                f"gt/y_{bn[0]}",
                y[0],
                current_step,
                self.audio_config.output_sampling_rate,
            )
            self.logger.experiment.add_figure(
                f"gt/y_spec_{bn[0]}",
                plot_spectrogram(x[0].cpu().numpy()),
                current_step,
            )
        #
        if batch_idx == 0:
            self.logger.experiment.add_audio(
                f"generated/y_hat_{bn[0]}",
                generated_wav[0],
                current_step,
                self.audio_config.output_sampling_rate,
            )

            y_hat_spec = dynamic_range_compression_torch(
                self.spectral_transform(generated_wav[0]).squeeze(1)[:, :, 1:]
            )
            self.logger.experiment.add_figure(
                f"generated/y_hat_spec_{bn[0]}",
                plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()),
                current_step,
            )

        self.log(
            "validation/mel_spec_error", val_err_tot, prog_bar=True, sync_dist=True
        )
