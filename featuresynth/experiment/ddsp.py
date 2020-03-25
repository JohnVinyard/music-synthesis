from ..audio import RawAudio
from .filterbank import FilterBankDiscriminator
from .realmelgan import Discriminator
from ..discriminator import ComplextSTFTDiscriminator
from ..discriminator.multiscale import MultiScaleMultiResDiscriminator
from ..generator.full import DDSPGenerator
from .experiment import Experiment
from ..loss import \
    mel_gan_disc_loss, mel_gan_gen_loss, least_squares_disc_loss, \
    least_squares_generator_loss
from .init import weights_init
from ..feature import spectrogram, audio
import zounds


class OneDimDDSPExperiment(Experiment):
    """
    The intution here is an extension of ideas from

    DDSP: Differentiable Digital Signal Processing
    https://arxiv.org/abs/2001.04643

    In short, we'll model sound as the sum of some number of sine oscillators
    as well as filtered/shaped white noise

    This model produces some of the clearest speech so far, but is brittle in
    that I *think* it strongly depends on its oscillators being placed along the
    same frequency axis as the input feature/spectrogram.  Put another way,
    it is largely just copying input to output.

    It is also plagued by
        - a lack of realistic higher-frequency content
        - Phase issues and issues with unrealistic harmonics
        - frenetic noise component that sounds like panting or frenzied clicking
    """
    def __init__(self):
        n_mels = 128
        n_fft = 1024
        hop = 256
        samplerate = zounds.SR22050()
        feature_size = 32
        total_samples = 8192

        n_osc = 128
        scale = zounds.MelScale(
            zounds.FrequencyBand(20, samplerate.nyquist - 20), n_osc)

        filter_bank = zounds.learn.FilterBank(
            samplerate, 511, scale, 0.9, normalize_filters=True,
            a_weighting=False)

        super().__init__(
            generator=DDSPGenerator(
                n_osc=n_osc,
                input_size=feature_size,
                in_channels=n_mels,
                output_size=total_samples,
                scale=scale,
                samplerate=samplerate),
            discriminator=MultiScaleMultiResDiscriminator(
                total_samples,
                flatten_multiscale_features=False,
                decompose=True,
                channel_judgements=True,
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
                'audio': (audio, (samplerate,)),
                'spectrogram': (spectrogram, (samplerate,))
            },
            total_samples=total_samples,
            feature_channels=n_mels,
            inference_sequence_factor=4,
            samplerate=samplerate)