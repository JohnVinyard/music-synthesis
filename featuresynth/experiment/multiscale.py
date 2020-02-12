from ..audio import RawAudio
from ..discriminator.multiscale import MultiScaleDiscriminator
from ..generator.multiscale import MultiScaleGenerator
from .experiment import Experiment
from ..util.modules import least_squares_disc_loss, least_squares_generator_loss
from ..feature import feature_channels


class MultiScaleExperiment(Experiment):
    def __init__(self):
        feature_size = 64
        super().__init__(
            generator=MultiScaleGenerator(feature_channels),
            discriminator=MultiScaleDiscriminator(),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=RawAudio,
            generator_loss=least_squares_generator_loss,
            discriminator_loss=least_squares_disc_loss)

