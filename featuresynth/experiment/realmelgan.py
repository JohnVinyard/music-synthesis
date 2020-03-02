import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np

from ..audio import RawAudio
from .experiment import Experiment
from ..loss import mel_gan_disc_loss, hinge_generator_loss
from .init import basic_init
from ..feature import audio, spectrogram
import zounds


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_size, ngf, n_residual_layers):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model


    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1,
                                       count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        # results = []
        features = []
        judgements = []
        for key, disc in self.model.items():
            z = disc(x)
            features.append(z[:-1])
            judgements.append(z[-1])
            # results.append(disc(x))
            x = self.downsample(x)
        return features, judgements


# loss_feat = 0
# feat_weights = 4.0 / (args.n_layers_D + 1)
# D_weights = 1.0 / args.num_D
# wt = D_weights * feat_weights
# for i in range(args.num_D):
#     for j in range(len(D_fake[i]) - 1):
#         loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())


def real_mel_gan_feature_loss(real_features, fake_features):
    loss = 0
    # features are lists of lists

    # scale by the number of discriminators
    # nd = (1 / len(real_features))
    feat_weights = 4.0 / (4 + 1)
    d_weights = 1 / 3
    wt = d_weights * feat_weights
    for r_group, f_group in zip(real_features, fake_features):

        # also scale by the number of layers in this discriminator
        # nl = (1 / len(r_group))

        for r_f, f_f in zip(r_group, f_group):
            l_loss = wt * F.l1_loss(r_f, f_f)
            loss += l_loss

    return loss


def mel_gan_gen_loss(
        real_features,
        fake_features,
        real_judgements,
        fake_judgements,
        gan_loss=hinge_generator_loss,
        feature_loss_weight=10):

    j_loss = sum(gan_loss(f) for r, f in zip(real_judgements, fake_judgements))

    f_loss = real_mel_gan_feature_loss(real_features, fake_features)
    return j_loss + (feature_loss_weight * f_loss)


def no_init(name, weight):
    pass


class RealMelGanExperiment(Experiment):
    def __init__(self):
        n_mels = 80
        size = 32
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192

        super().__init__(
            Generator(n_mels, size, n_residual_layers=3),
            Discriminator(num_D=3, ndf=16, n_layers=4, downsampling_factor=4),
            learning_rate=1e-4,
            feature_size=size,
            audio_repr_class=RawAudio,
            generator_loss=mel_gan_gen_loss,
            discriminator_loss=mel_gan_disc_loss,
            g_init=no_init,
            d_init=no_init,
            feature_funcs={
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate, n_fft, hop, n_mels))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            samplerate=samplerate
        )


