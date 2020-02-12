from ..audio import MDCT
from ..discriminator import MDCTDiscriminator, TwoDimMDCTDiscriminator
from ..feature import feature_channels
from ..generator import \
    MDCTGenerator, TwoDimMDCTGenerator, UnconditionedGenerator

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


class TwoDimMDCTExperiment(Experiment):
    """
    The intuition here is that a two-dimensional generator will be better able
    to generalize to frequency-transposed shapes

    This generator is able to overfit with a good deal of noise

    The generations don't seem much better than the 1D generator
    """

    def __init__(self):
        super().__init__(
            generator=TwoDimMDCTGenerator(feature_channels),
            discriminator=MDCTDiscriminator(feature_channels),
            learning_rate=1e-4,
            feature_size=64,
            audio_repr_class=MDCT,
            generator_loss=least_squares_generator_loss,
            discriminator_loss=least_squares_disc_loss)


class TwoDimMDCTDiscriminatorExperiment(Experiment):
    """
    Intuition:
        Perhaps the responsibility for capturing translation-invariant
        structures lies more in the discriminator's domain?

    The generator is able to overfit with a good deal of noise

    Generations aren't much better than the original 1D generator
    """

    def __init__(self):
        super().__init__(
            generator=MDCTGenerator(feature_channels),
            discriminator=MDCTDiscriminator(feature_channels),
            learning_rate=1e-4,
            feature_size=64,
            audio_repr_class=MDCT,
            generator_loss=least_squares_generator_loss,
            discriminator_loss=least_squares_disc_loss)


class FullTwoDimMDCTDiscriminatorExperiment(Experiment):
    """
    Intuition:
        Is it helpful if both generator and discriminator have 2d layers?


    Overfitting:
        Overfits with significant noise


    Generations are worse than original 1D generator
    """

    def __init__(self):
        super().__init__(
            generator=TwoDimMDCTGenerator(feature_channels),
            discriminator=MDCTDiscriminator(feature_channels),
            learning_rate=1e-4,
            feature_size=64,
            audio_repr_class=MDCT,
            generator_loss=least_squares_generator_loss,
            discriminator_loss=least_squares_disc_loss)


class UnconditionedGeneratorExperiment(Experiment):
    """
    Intuition:
        The projection from mel spectrogram to the linear frequency space of
        the MDCT representation is making it difficult to produce convincingly
        real spectrograms.  Try the classic, unconditioned spectrogram to see
        what changes.

    Overfitting:
        The generator can produce convincing voice texture, with a good deal of
        noise, however, unlike the conditioned generator, it produces
        stuttering, repeating and echoic speech, likely due to the patch-based
        discriminator.

    Conclusion:
        Overall, my hypothesis that the conditioning input is causing problems
        doesn't seem to be the case.  These generations are significantly worse
        and less realistic
    """
    def __init__(self):
        super().__init__(
            generator=UnconditionedGenerator(),
            discriminator=TwoDimMDCTDiscriminator(),
            learning_rate=1e-4,
            feature_size=64,
            audio_repr_class=MDCT,
            generator_loss=least_squares_generator_loss,
            discriminator_loss=least_squares_disc_loss)
