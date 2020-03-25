import zounds

from .experiment import Experiment
from .init import weights_init
from ..audio import MDCT
from ..discriminator import MDCTDiscriminator
from ..feature import audio, spectrogram
from ..generator import MDCTGenerator, GroupedMDCTGenerator
from ..loss import \
    mel_gan_disc_loss, mel_gan_gen_loss, least_squares_generator_loss, \
    least_squares_disc_loss



class GroupedMDCTExperiment(Experiment):

    def __init__(self):
        feature_channels = 128
        feature_size = 32
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192


        super().__init__(
            generator=GroupedMDCTGenerator(feature_channels),
            discriminator=MDCTDiscriminator(
                MDCT.mdct_bins(),
                feature_size,
                conditioning_channels=feature_channels),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=MDCT,
            generator_loss=mel_gan_gen_loss,
            sub_gen_loss=least_squares_generator_loss,
            discriminator_loss=mel_gan_disc_loss,
            sub_disc_loss=least_squares_disc_loss,
            g_init=weights_init,
            d_init=weights_init,
            feature_funcs={
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate,))
            },
            total_samples=total_samples,
            feature_channels=feature_channels,
            samplerate=samplerate,
            inference_sequence_factor=4)