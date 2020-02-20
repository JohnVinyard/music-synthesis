from ..audio import RawAudio
from ..discriminator.full import FullDiscriminator
from ..discriminator import MelGanDiscriminator
from ..generator.full import TwoDimDDSPGenerator, DDSPGenerator
from .experiment import Experiment
from ..feature import feature_channels
from ..loss import mel_gan_disc_loss, mel_gan_gen_loss


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
        feature_size = 64 + 32
        super().__init__(
            generator=DDSPGenerator(feature_size, feature_channels),
            discriminator=MelGanDiscriminator(),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=mel_gan_gen_loss,
            discriminator_loss=mel_gan_disc_loss)




class DDSPExperiment(Experiment):
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
        feature_size = 64 + 32
        super().__init__(
            generator=TwoDimDDSPGenerator(feature_size, feature_channels),
            discriminator=MelGanDiscriminator(),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=mel_gan_gen_loss,
            discriminator_loss=mel_gan_disc_loss)



