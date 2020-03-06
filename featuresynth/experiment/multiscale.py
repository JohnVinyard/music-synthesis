from ..audio import RawAudio, MultiScale
from ..discriminator.multiscale import MultiScaleMultiResDiscriminator
from ..generator.multiscale import MultiScaleGenerator
from .experiment import Experiment
from ..loss import \
    mel_gan_gen_loss, mel_gan_disc_loss, least_squares_generator_loss, \
    least_squares_disc_loss
import zounds
from ..feature import normalized_and_augmented_audio, make_spectrogram_func
from .init import weights_init

"""
Things To Try:
- judgements per band in addition to top-level judgement
- Filter bank as first discriminator layer and last generator layer for each channel
- audio representation where bands are stored separately rather than being
  resampled and summed together for better gradients?
"""




class MultiScaleMultiResGroupedFeaturesExperiment(Experiment):
    """

    """

    def __init__(self):
        n_mels = 128
        feature_size = 32
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192

        spec_func = make_spectrogram_func(
            normalized_and_augmented_audio, samplerate, n_fft, hop, n_mels)

        super().__init__(
            generator=MultiScaleGenerator(
                n_mels, feature_size, total_samples, transposed_conv=True),
            discriminator=MultiScaleMultiResDiscriminator(
                total_samples, flatten_multiscale_features=True),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
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
            feature_channels=n_mels,
            samplerate=samplerate,
            inference_sequence_factor=4)



class MultiScaleNoDeRecompose(Experiment):
    """

    """

    def __init__(self):
        n_mels = 128
        feature_size = 32
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192

        spec_func = make_spectrogram_func(
            normalized_and_augmented_audio, samplerate, n_fft, hop, n_mels)

        super().__init__(
            generator=MultiScaleGenerator(
                n_mels,
                feature_size,
                total_samples,
                transposed_conv=True,
                recompose=False),
            discriminator=MultiScaleMultiResDiscriminator(
                total_samples,
                flatten_multiscale_features=True,
                decompose=False),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=MultiScale,
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
            feature_channels=n_mels,
            samplerate=samplerate,
            inference_sequence_factor=4)