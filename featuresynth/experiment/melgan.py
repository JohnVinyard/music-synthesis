from ..audio import RawAudio
from ..discriminator.full import FullDiscriminator
from ..feature import feature_channels
from ..generator.full import MelGanGenerator
from .experiment import Experiment
from ..util.modules import least_squares_disc_loss, least_squares_generator_loss


class MelGanExperiment(Experiment):
    def __init__(self):
        feature_size = 64
        super().__init__(
            generator=MelGanGenerator(feature_size, feature_channels),
            discriminator=FullDiscriminator(),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=least_squares_generator_loss,
            discriminator_loss=least_squares_disc_loss)

