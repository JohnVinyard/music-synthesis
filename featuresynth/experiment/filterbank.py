from ..audio import RawAudio
from ..discriminator import \
    FilterBankDiscriminator, LargeReceptiveFieldFilterBankDiscriminator
from ..generator import FilterBankGenerator
from .experiment import Experiment
from ..util.modules import least_squares_disc_loss, least_squares_generator_loss
from ..feature import sr
import zounds

scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), 128)
filter_bank = zounds.learn.FilterBank(
    sr, 511, scale, 0.9, normalize_filters=True, a_weighting=False)

"""
https://openreview.net/pdf?id=9jTbNbBNw0
Things to try:

- larger receptive field in generator
- residual blocks in generator
- max pooling instead of average pooling in discriminator
- different padding (reflection, replication)
- conditioned discriminator
"""


class LargeReceptiveFieldFilterBankExperiment(Experiment):
    def __init__(self):
        feature_size = 64
        super().__init__(
            generator=FilterBankGenerator(filter_bank),
            discriminator=LargeReceptiveFieldFilterBankDiscriminator(
                filter_bank),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=least_squares_generator_loss,
            discriminator_loss=least_squares_disc_loss)


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
            generator_loss=least_squares_generator_loss,
            discriminator_loss=least_squares_disc_loss)
