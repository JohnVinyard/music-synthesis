from ..audio import RawAudio
from ..discriminator import FullDiscriminator, MelGanDiscriminator
from ..generator.full import MelGanGenerator
from .experiment import Experiment
from ..loss import mel_gan_disc_loss, mel_gan_gen_loss
from .init import basic_init, discriminator_init
from ..feature import audio, spectrogram
import zounds


class MultiScaleMelGanExperiment(Experiment):
    """
    This produces pretty good sounding audio after about 12 hours, but still.
    The speech sounds realistic, but is hard to understand.  No audible phase
    issues.
    """

    def __init__(self):
        total_samples = 8192
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        n_mels = 128
        feature_size = total_samples // hop

        super().__init__(
            generator=MelGanGenerator(feature_size, n_mels),
            discriminator=MelGanDiscriminator(),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=mel_gan_gen_loss,
            discriminator_loss=mel_gan_disc_loss,
            g_init=basic_init,
            d_init=discriminator_init,
            feature_funcs={
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate, n_fft, hop, n_mels))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            samplerate=samplerate)


class MelGanExperiment(Experiment):
    def __init__(self):
        feature_size = 64
        super().__init__(
            generator=MelGanGenerator(feature_size, feature_channels),
            discriminator=FullDiscriminator(),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=mel_gan_gen_loss,
            discriminator_loss=mel_gan_disc_loss)
