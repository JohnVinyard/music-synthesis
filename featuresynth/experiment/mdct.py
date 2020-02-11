from ..audio import MDCT
from ..discriminator import MDCTDiscriminator
from ..feature import feature_channels
from ..generator import MDCTGenerator

from ..util.modules import least_squares_disc_loss, least_squares_generator_loss

from .experiment import Experiment


class MDCTExperiment(Experiment):
    def __init__(self):
        super().__init__(
            generator=MDCTGenerator(feature_channels),
            discriminator=MDCTDiscriminator(feature_channels),
            learning_rate=1e-4,
            feature_size=64,
            audio_repr_class=MDCT,
            generator_loss=least_squares_generator_loss,
            discriminator_loss=least_squares_disc_loss)
