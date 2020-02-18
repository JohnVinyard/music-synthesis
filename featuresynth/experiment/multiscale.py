from ..audio import RawAudio
from ..discriminator.multiscale import MultiScaleDiscriminator
from ..generator.multiscale import MultiScaleGenerator
from .experiment import Experiment
from ..feature import feature_channels
from ..loss import mel_gan_gen_loss, mel_gan_disc_loss


class MultiScaleExperiment(Experiment):
    """
    The intuition here is that when producing audio at a given sample rate,
    many/most "important" frequency bands require a much lower sampling rate.

    Here, we split the generator into five bands that generate audio at
    different rates, then upsample and combine the results.

    The discriminator performs the same operation in reverse, decomposing and
    resampling the input into five bands sampled at appropriate rates, analyzes
    each band and then combines the results

    This approach shows promise, and deserves an overnight run
    """
    def __init__(self):
        feature_size = 64
        super().__init__(
            generator=MultiScaleGenerator(feature_channels),
            discriminator=MultiScaleDiscriminator(),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=mel_gan_gen_loss,
            discriminator_loss=mel_gan_disc_loss)

