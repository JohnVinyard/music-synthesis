from ..audio import RawAudio, MultiScale
from ..discriminator.multiscale import MultiScaleMultiResDiscriminator, FilterBankMultiScaleDiscriminator
from ..discriminator.fft import ComplextSTFTDiscriminator
from ..generator.multiscale import MultiScaleGenerator, FilterBankMultiScaleGenerator
from ..generator.full import DDSPGenerator
from ..generator.fft import ComplextSTFTGenerator
from .experiment import Experiment
from ..loss import \
    mel_gan_gen_loss, mel_gan_disc_loss, least_squares_generator_loss, \
    least_squares_disc_loss
import zounds
from ..feature import audio, spectrogram
from .init import weights_init

"""
Things To Try:
- judgements per band in addition to top-level judgement
- Filter bank as first discriminator layer and last generator layer for each channel
- audio representation where bands are stored separately rather than being
  resampled and summed together for better gradients?
"""


class FilterBankMultiscaleExperiment(Experiment):

    AUDIO_REPR_CLASS = MultiScale
    SAMPLERATE = zounds.SR22050()
    N_MELS = 128
    feature_size = 32
    total_samples = 8192

    @classmethod
    def make_generator(cls):
        return FilterBankMultiScaleGenerator(
            cls.SAMPLERATE,
            cls.N_MELS,
            cls.feature_size,
            cls.total_samples,
            recompose=False)

    def __init__(self):
        super().__init__(
            generator=FilterBankMultiScaleGenerator(
                self.SAMPLERATE,
                self.N_MELS,
                self.feature_size,
                self.total_samples,
                recompose=False),
            discriminator=FilterBankMultiScaleDiscriminator(
                self.total_samples,
                self.SAMPLERATE,
                decompose=False,
                conditioning_channels=self.N_MELS),
            learning_rate=1e-4,
            feature_size=self.feature_size,
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
            total_samples=self.total_samples,
            feature_channels=self.N_MELS,
            samplerate=self.SAMPLERATE,
            inference_sequence_factor=4)


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


        super().__init__(
            generator=MultiScaleGenerator(
                n_mels,
                feature_size,
                total_samples,
                transposed_conv=True,
                recompose=True),
            discriminator=MultiScaleMultiResDiscriminator(
                total_samples,
                flatten_multiscale_features=False,
                channel_judgements=True,
                conditioning_channels=n_mels,
                decompose=True),
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
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate,))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            samplerate=samplerate,
            inference_sequence_factor=4)


class MultiScaleNoDeRecomposeUnconditionedShortKernel(Experiment):
    """

    """

    def __init__(self):
        n_mels = 128
        feature_size = 32
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192


        super().__init__(
            generator=MultiScaleGenerator(
                n_mels,
                feature_size,
                total_samples,
                transposed_conv=True,
                recompose=False),
            discriminator=MultiScaleMultiResDiscriminator(
                total_samples,
                flatten_multiscale_features=False,
                channel_judgements=True,
                decompose=False,
                kernel_size=9),
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
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate,))
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


        super().__init__(
            generator=MultiScaleGenerator(
                n_mels,
                feature_size,
                total_samples,
                transposed_conv=True,
                recompose=False),
            discriminator=MultiScaleMultiResDiscriminator(
                total_samples,
                flatten_multiscale_features=False,
                channel_judgements=True,
                conditioning_channels=n_mels,
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
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate,))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            samplerate=samplerate,
            inference_sequence_factor=4)


class MultiScaleWithDeRecompose(Experiment):
    """

    """

    def __init__(self):
        n_mels = 128
        feature_size = 32
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192


        super().__init__(
            generator=MultiScaleGenerator(
                n_mels,
                feature_size,
                total_samples,
                transposed_conv=True,
                recompose=True),
            discriminator=MultiScaleMultiResDiscriminator(
                total_samples,
                flatten_multiscale_features=False,
                channel_judgements=True,
                conditioning_channels=n_mels,
                decompose=True,
                kernel_size=9),
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
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate,))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            samplerate=samplerate,
            inference_sequence_factor=4)


class MultiScaleNoDeRecomposeShortKernels(Experiment):
    """

    """

    def __init__(self):
        n_mels = 128
        feature_size = 32
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192


        super().__init__(
            generator=MultiScaleGenerator(
                n_mels,
                feature_size,
                total_samples,
                transposed_conv=True,
                recompose=False,
                kernel_size=8),
            discriminator=MultiScaleMultiResDiscriminator(
                total_samples,
                flatten_multiscale_features=False,
                channel_judgements=True,
                conditioning_channels=n_mels,
                decompose=False,
                kernel_size=9),
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
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate,))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            samplerate=samplerate,
            inference_sequence_factor=4)



class MultiScaleWithSTFTDiscriminator(Experiment):
    """

    """

    def __init__(self):
        n_mels = 128
        feature_size = 32
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192


        super().__init__(
            generator=MultiScaleGenerator(
                n_mels,
                feature_size,
                total_samples,
                transposed_conv=True,
                recompose=True),
            discriminator=ComplextSTFTDiscriminator(
                n_fft,
                hop,
                n_mels,
                do_fft=True),
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
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate,))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            samplerate=samplerate,
            inference_sequence_factor=4)



class MultiScaleWithDDSPGenerator(Experiment):
    """

    """

    def __init__(self):
        n_mels = 128
        feature_size = 32
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192

        n_osc = 128
        scale = zounds.LinearScale(
            zounds.FrequencyBand(20, samplerate.nyquist - 20), n_osc)
        super().__init__(
            generator=DDSPGenerator(
                n_osc,
                feature_size,
                n_mels,
                total_samples,
                scale,
                samplerate),
            discriminator=MultiScaleMultiResDiscriminator(
                total_samples,
                flatten_multiscale_features=False,
                channel_judgements=True,
                conditioning_channels=n_mels,
                decompose=True),
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
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate,))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            samplerate=samplerate,
            inference_sequence_factor=4)




class MultiScaleNoDeRecomposeNoConvTranspose(Experiment):
    """

    """

    def __init__(self):
        n_mels = 128
        feature_size = 32
        samplerate = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192


        super().__init__(
            generator=MultiScaleGenerator(
                n_mels,
                feature_size,
                total_samples,
                transposed_conv=False,
                recompose=False),
            discriminator=MultiScaleMultiResDiscriminator(
                total_samples,
                flatten_multiscale_features=False,
                channel_judgements=True,
                conditioning_channels=n_mels,
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
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate,))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            samplerate=samplerate,
            inference_sequence_factor=4)