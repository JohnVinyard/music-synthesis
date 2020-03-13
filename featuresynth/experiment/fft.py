from ..audio import ComplextSTFT
from ..generator import ComplextSTFTGenerator
from ..discriminator import ComplextSTFTDiscriminator
import zounds
from ..loss import \
    mel_gan_disc_loss, mel_gan_gen_loss, least_squares_generator_loss, \
    least_squares_disc_loss
from .init import weights_init
from ..feature import audio, spectrogram

from .experiment import Experiment


class ComplexSTFTExperiment(Experiment):

    N_MELS = 128
    FEATURE_SIZE = 32
    SAMPLERATE = zounds.SR22050()
    N_FFT = 1024
    HOP = 256
    TOTAL_SAMPLES = 8192
    AUDIO_REPR_CLASS = ComplextSTFT

    @classmethod
    def make_generator(cls):
        return ComplextSTFTGenerator(cls.N_MELS, cls.N_FFT, cls.HOP)

    def __init__(self):
        super().__init__(
            generator=self.make_generator(),
            discriminator=ComplextSTFTDiscriminator(
                window_size=self.N_FFT,
                hop=self.HOP,
                conditioning_channels=self.N_MELS),
            learning_rate=1e-4,
            feature_size=self.FEATURE_SIZE,
            audio_repr_class=self.AUDIO_REPR_CLASS,
            generator_loss=mel_gan_gen_loss,
            sub_gen_loss=least_squares_generator_loss,
            discriminator_loss=mel_gan_disc_loss,
            sub_disc_loss=least_squares_disc_loss,
            g_init=weights_init,
            d_init=weights_init,
            feature_funcs={
                'audio': (audio, (self.SAMPLERATE,)),
                'spectrogram': (spectrogram, (self.SAMPLERATE,))
            },
            total_samples=self.TOTAL_SAMPLES,
            feature_channels=self.N_MELS,
            samplerate=self.SAMPLERATE,
            inference_sequence_factor=4)
