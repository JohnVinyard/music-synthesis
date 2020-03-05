from ..audio import MDCT
from ..discriminator import MDCTDiscriminator, TwoDimMDCTDiscriminator
from ..generator import \
    MDCTGenerator, TwoDimMDCTGenerator, UnconditionedGenerator, \
    GroupedMDCTGenerator
import zounds
from ..loss import mel_gan_disc_loss, mel_gan_gen_loss
from .init import weights_init
from ..feature import make_spectrogram_func, normalized_and_augmented_audio

from .experiment import Experiment


# class MDCTExperiment(Experiment):
#     """
#     This learns the large scale structure of the speech pretty well, but never
#     really learns to produce tones or harmonics, resulting in scratchy, static-y
#     generations
#     """
#
#     def __init__(self):
#         feature_channels = 256
#         super().__init__(
#             generator=MDCTGenerator(feature_channels),
#             discriminator=MDCTDiscriminator(feature_channels),
#             learning_rate=1e-4,
#             feature_size=64,
#             audio_repr_class=MDCT,
#             generator_loss=mel_gan_gen_loss,
#             discriminator_loss=mel_gan_disc_loss)


class GroupedMDCTExperiment(Experiment):

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
            generator=MDCTGenerator(feature_channels),
            discriminator=MDCTDiscriminator(MDCT.mdct_bins(), feature_size),
            learning_rate=1e-4,
            feature_size=feature_size,
            audio_repr_class=MDCT,
            generator_loss=mel_gan_gen_loss,
            discriminator_loss=mel_gan_disc_loss,
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


# class TwoDimMDCTExperiment(Experiment):
#     """
#     The intuition here is that a two-dimensional generator will be better able
#     to generalize to frequency-transposed shapes
#
#     This generator is able to overfit with a good deal of noise
#
#     The generations don't seem much better than the 1D generator
#     """
#
#     def __init__(self):
#         feature_channels = 256
#         super().__init__(
#             generator=TwoDimMDCTGenerator(feature_channels),
#             discriminator=MDCTDiscriminator(feature_channels),
#             learning_rate=1e-4,
#             feature_size=64,
#             audio_repr_class=MDCT,
#             generator_loss=mel_gan_gen_loss,
#             discriminator_loss=mel_gan_disc_loss)
#
#
# class TwoDimMDCTDiscriminatorExperiment(Experiment):
#     """
#     Intuition:
#         Perhaps the responsibility for capturing translation-invariant
#         structures lies more in the discriminator's domain?
#
#     The generator is able to overfit with a good deal of noise
#
#     Generations aren't much better than the original 1D generator
#     """
#
#     def __init__(self):
#         feature_channels = 256
#         super().__init__(
#             generator=MDCTGenerator(feature_channels),
#             discriminator=MDCTDiscriminator(feature_channels),
#             learning_rate=1e-4,
#             feature_size=64,
#             audio_repr_class=MDCT,
#             generator_loss=mel_gan_gen_loss,
#             discriminator_loss=mel_gan_disc_loss)
#
#
# class FullTwoDimMDCTDiscriminatorExperiment(Experiment):
#     """
#     Intuition:
#         Is it helpful if both generator and discriminator have 2d layers?
#
#
#     Overfitting:
#         Overfits with significant noise
#
#
#     Generations are worse than original 1D generator
#     """
#
#     def __init__(self):
#         feature_channels = 256
#         super().__init__(
#             generator=TwoDimMDCTGenerator(feature_channels),
#             discriminator=MDCTDiscriminator(feature_channels),
#             learning_rate=1e-4,
#             feature_size=64,
#             audio_repr_class=MDCT,
#             generator_loss=mel_gan_gen_loss,
#             discriminator_loss=mel_gan_disc_loss)
#
#
# class UnconditionedGeneratorExperiment(Experiment):
#     """
#     Intuition:
#         The projection from mel spectrogram to the linear frequency space of
#         the MDCT representation is making it difficult to produce convincingly
#         real spectrograms.  Try the classic, unconditioned spectrogram to see
#         what changes.
#
#     Overfitting:
#         The generator can produce convincing voice texture, with a good deal of
#         noise, however, unlike the conditioned generator, it produces
#         stuttering, repeating and echoic speech, likely due to the patch-based
#         discriminator.
#
#     Conclusion:
#         Overall, my hypothesis that the conditioning input is causing problems
#         doesn't seem to be the case.  These generations are significantly worse
#         and less realistic
#     """
#
#     def __init__(self):
#         super().__init__(
#             generator=UnconditionedGenerator(),
#             discriminator=TwoDimMDCTDiscriminator(),
#             learning_rate=1e-4,
#             feature_size=64,
#             audio_repr_class=MDCT,
#             generator_loss=mel_gan_gen_loss,
#             discriminator_loss=mel_gan_disc_loss)
