from ..audio import RawAudio
from ..discriminator import \
    FilterBankDiscriminator, LargeReceptiveFieldFilterBankDiscriminator
from ..generator import FilterBankGenerator, ResidualStackFilterBankGenerator
from .experiment import Experiment
from ..loss import mel_gan_gen_loss, mel_gan_disc_loss
import zounds
from .init import weights_init
from ..feature import \
    normalized_and_augmented_audio, make_spectrogram_func, audio, spectrogram
from ..loss import least_squares_disc_loss, least_squares_generator_loss


"""
https://openreview.net/pdf?id=9jTbNbBNw0
Things to try:

- larger receptive field in discriminator - Less long range coherence
- residual blocks in generator - Still doesn't converge to meaningful speech
- just low resolution - this results in more long ran
- max pooling instead of average pooling in discriminator
- different padding (reflection, replication)
- conditioned discriminator
"""





class FilterBankExperiment(Experiment):
    """
    This is probably the best audio quality yet.  The audio is relatively
    crisp, spectrograms are indistinguishable from real speech, although they
    are hard to understand.

    There are definite phase issues here and there after 12 hours.

    Overall, the texture of the speech is more realistic than what's produced
    by the basic MelGAN setup.
    """

    N_MELS = 128
    FEATURE_SIZE = 32
    SAMPLERATE = zounds.SR22050()
    N_FFT = 1024
    HOP = 256
    TOTAL_SAMPLES = 8192

    @classmethod
    def make_filter_bank(cls, samplerate):
        scale = zounds.LinearScale(
            zounds.FrequencyBand(20, samplerate.nyquist - 20), 128)
        filter_bank = zounds.learn.FilterBank(
            samplerate,
            511,
            scale,
            0.9,
            normalize_filters=True,
            a_weighting=False)
        return filter_bank


    @classmethod
    def make_generator(cls, filter_bank=None):
        filter_bank = filter_bank or cls.make_filter_bank(cls.SAMPLERATE)
        return FilterBankGenerator(
            filter_bank, cls.FEATURE_SIZE, cls.TOTAL_SAMPLES, cls.N_MELS)


    @classmethod
    def make_spec_func(cls):
        return make_spectrogram_func(
            normalized_and_augmented_audio,
            cls.SAMPLERATE,
            cls.N_FFT,
            cls.HOP,
            cls.N_MELS)

    def __init__(self):
        # n_mels = 128
        # feature_size = 32
        # sr = zounds.SR22050()
        # n_fft = 1024
        # hop = 256
        # total_samples = 8192

        # scale = zounds.LinearScale(
        #     zounds.FrequencyBand(20, sr.nyquist - 20), 128)
        # filter_bank = zounds.learn.FilterBank(
        #     sr, 511, scale, 0.9, normalize_filters=True, a_weighting=False)

        filter_bank = self.make_filter_bank(self.SAMPLERATE)

        # spec_func = make_spectrogram_func(
        #     normalized_and_augmented_audio, sr, n_fft, hop, n_mels)

        # spec_func = make_spectrogram_func(
        #     normalized_and_augmented_audio,
        #     self.SAMPLERATE,
        #     self.N_FFT,
        #     self.HOP,
        #     self.N_MELS)

        # spec_func = self.make_spec_func()

        super().__init__(
            # generator=FilterBankGenerator(
            #     filter_bank, feature_size, total_samples, n_mels),
            generator=self.make_generator(),
            # discriminator=FilterBankDiscriminator(filter_bank, total_samples),
            discriminator=FilterBankDiscriminator(filter_bank, self.TOTAL_SAMPLES),
            learning_rate=1e-4,
            # feature_size=feature_size,
            feature_size=self.FEATURE_SIZE,
            audio_repr_class=RawAudio,
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


class ConditionalFilterBankExperiment(Experiment):
    """
    """

    def __init__(self):
        n_mels = 128
        feature_size = 32
        sr = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192

        scale = zounds.LinearScale(
            zounds.FrequencyBand(20, sr.nyquist - 20), 128)
        filter_bank = zounds.learn.FilterBank(
            sr, 511, scale, 0.9, normalize_filters=True, a_weighting=False)

        spec_func = make_spectrogram_func(
            normalized_and_augmented_audio, sr, n_fft, hop, n_mels)

        super().__init__(
            generator=FilterBankGenerator(
                filter_bank, feature_size, total_samples, n_mels),
            discriminator=FilterBankDiscriminator(
                filter_bank,
                total_samples,
                conditioning_channels=n_mels),
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
                'audio': (normalized_and_augmented_audio, (sr,)),
                'spectrogram': (spec_func, (sr,))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            samplerate=sr,
            inference_sequence_factor=4)


class   AlternateFilterBankExperiment(Experiment):
    """
    This is very similar to the general FilterBank, with a several differences:
        - The architecure models the original MelGAN experiment
        - weight norm is used
        - the generator produces both a harmonic (via filter banks) and noise
          component
        - the generator uses a linear scale while the discriminator uses a mel
          scale

    There's definitely promise in this model, but I've only trained it for
    ~8 hours with a very small batch size (2) due to my GPU memory constraints.
    It's definitely worth further exploration.

    For now generations can be heard here:

    https://generation-report-alternatefilterbankexperiment.s3-us-west-1.amazonaws.com/index.html

    It's definitely worth trying out a linear scale in the discriminator as well
    as these generations demonstrate some of the muffled quality of earlier
    filter bank generators before switching to linear scales.
    """

    def __init__(self):
        n_mels = 128
        feature_size = 32
        sr = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192


        freq_band = zounds.FrequencyBand(20, sr.nyquist - 20)
        n_filters = 128
        filter_taps = 511

        gen_scale = zounds.LinearScale(freq_band, n_filters)
        gen_filter_bank = zounds.learn.FilterBank(
            sr,
            filter_taps,
            gen_scale,
            0.9,
            normalize_filters=True,
            a_weighting=False)

        disc_scale = zounds.MelScale(freq_band, n_filters)
        disc_filter_bank = zounds.learn.FilterBank(
            sr,
            filter_taps,
            disc_scale,
            0.9,
            normalize_filters=True,
            a_weighting=False)

        spec_func = make_spectrogram_func(
            normalized_and_augmented_audio, sr, n_fft, hop, n_mels)

        super().__init__(
            generator=ResidualStackFilterBankGenerator(
                gen_filter_bank,
                feature_size,
                total_samples,
                n_mels,
                add_weight_norm=True),
            discriminator=LargeReceptiveFieldFilterBankDiscriminator(
                disc_filter_bank,
                add_weight_norm=True),
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
                'audio': (normalized_and_augmented_audio, (sr,)),
                'spectrogram': (spec_func, (sr,))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            samplerate=sr,
            inference_sequence_factor=4)
