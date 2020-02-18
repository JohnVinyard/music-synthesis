from ..audio import RawAudio
from ..discriminator import \
    FilterBankDiscriminator, LargeReceptiveFieldFilterBankDiscriminator, \
    LowResFilterBankDiscriminator
from ..generator import FilterBankGenerator, ResidualStackFilterBankGenerator
from .experiment import Experiment
from ..feature import sr
from ..loss import mel_gan_gen_loss, mel_gan_disc_loss
import zounds

scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), 128)
filter_bank = zounds.learn.FilterBank(
    sr, 511, scale, 0.9, normalize_filters=True, a_weighting=False)

"""
https://openreview.net/pdf?id=9jTbNbBNw0
Things to try:

- larger receptive field in discriminator - Less long range coherence
- residual blocks in generator - Still doesn't converge to meaningful speech
- just low resolution
- max pooling instead of average pooling in discriminator
- different padding (reflection, replication)
- conditioned discriminator
"""


class LowResFilterBankExperiment(Experiment):
    def __init__(self):
        feature_size = 64
        super().__init__(
            generator=FilterBankGenerator(filter_bank),
            discriminator=LowResFilterBankDiscriminator(filter_bank),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=mel_gan_gen_loss,
            discriminator_loss=mel_gan_disc_loss)


class ResidualStackFilterBankExperiment(Experiment):
    """
    This ends up with worse generations than the original vanilla
    FilterBankExperiment
    """
    def __init__(self):
        feature_size = 64
        super().__init__(
            generator=ResidualStackFilterBankGenerator(filter_bank),
            discriminator=FilterBankDiscriminator(filter_bank),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=mel_gan_gen_loss,
            discriminator_loss=mel_gan_disc_loss)


class LargeReceptiveFieldFilterBankExperiment(Experiment):
    """
    Surprisingly, this experiment with larger receptive fields in the
    discriminator ends up with much *less* coherence, with jittery, short
    frames of realistic speech that form no coherent larger-scale picture.
    """

    def __init__(self):
        feature_size = 64
        super().__init__(
            generator=FilterBankGenerator(filter_bank),
            discriminator=LargeReceptiveFieldFilterBankDiscriminator(
                filter_bank),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=mel_gan_gen_loss,
            discriminator_loss=mel_gan_disc_loss)


class FilterBankExperiment(Experiment):
    """
    This is probably the best audio quality yet.  The audio is relatively
    crisp, phase is mostly coherent and spectrograms are indistinguishable from
    real speech.

    There is almost no long-term coherence, however, and the generator produces
    realistic babbling without accurately reproducing the speech from the
    spectrograms.
    """

    def __init__(self):
        feature_size = 64
        super().__init__(
            generator=FilterBankGenerator(filter_bank),
            discriminator=FilterBankDiscriminator(filter_bank),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=mel_gan_gen_loss,
            discriminator_loss=mel_gan_disc_loss)
