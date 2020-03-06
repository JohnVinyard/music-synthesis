from ..audio import ComplextSTFT
from ..generator import ComplextSTFTGenerator
from ..discriminator import ComplextSTFTDiscriminator
import zounds
from ..loss import \
    mel_gan_disc_loss, mel_gan_gen_loss, least_squares_generator_loss, \
    least_squares_disc_loss
from .init import weights_init
from ..feature import make_spectrogram_func, normalized_and_augmented_audio

from .experiment import Experiment

"""
Things to try:
- multi-head generator that constrains phase data to the range (-pi, pi)
"""


class ComplexSTFTExperiment(Experiment):

    def __init__(self):
        feature_channels = 128
        feature_size = 32
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192

        spec_func = make_spectrogram_func(
            normalized_and_augmented_audio,
            samplerate,
            n_fft,
            hop,
            feature_channels)


        super().__init__(
            generator=ComplextSTFTGenerator(feature_channels, n_fft, hop),
            discriminator=ComplextSTFTDiscriminator(n_fft, hop),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=ComplextSTFT,
            generator_loss=mel_gan_gen_loss,
            sub_gen_loss=least_squares_generator_loss,
            discriminator_loss=mel_gan_disc_loss,
            sub_disc_loss=least_squares_disc_loss,
            g_init=weights_init,
            d_init=weights_init,
            feature_funcs={
                'audio': (normalized_and_augmented_audio, (samplerate,)),
                'spectrogram': (spec_func, (samplerate,))
            },
            total_samples=total_samples,
            feature_channels=feature_channels,
            samplerate=samplerate,
            inference_sequence_factor=4)
