from ..audio import RawAudio
from ..discriminator import FilterBankDiscriminator
from ..generator import FilterBankGenerator
from .experiment import Experiment
from ..loss import mel_gan_gen_loss, mel_gan_disc_loss
import zounds
from .init import weights_init
from ..feature import normalized_and_augmented_audio, make_spectrogram_func
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



# class LowResFilterBankExperiment(Experiment):
#     def __init__(self):
#         feature_size = 64
#         sr = zounds.SR11025()
#         scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), 128)
#         filter_bank = zounds.learn.FilterBank(
#             sr, 511, scale, 0.9, normalize_filters=True, a_weighting=False)
#
#         super().__init__(
#             generator=FilterBankGenerator(filter_bank),
#             discriminator=LowResFilterBankDiscriminator(filter_bank),
#             learning_rate=1e-4,
#             feature_size=feature_size,
#             audio_repr_class=RawAudio,
#             generator_loss=mel_gan_gen_loss,
#             discriminator_loss=mel_gan_disc_loss)
#
#
# class ResidualStackFilterBankExperiment(Experiment):
#     """
#     This ends up with worse generations than the original vanilla
#     FilterBankExperiment
#     """
#     def __init__(self):
#         feature_size = 64
#         sr = zounds.SR11025()
#         scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), 128)
#         filter_bank = zounds.learn.FilterBank(
#             sr, 511, scale, 0.9, normalize_filters=True, a_weighting=False)
#         super().__init__(
#             generator=ResidualStackFilterBankGenerator(filter_bank),
#             discriminator=FilterBankDiscriminator(filter_bank),
#             learning_rate=1e-4,
#             feature_size=feature_size,
#             audio_repr_class=RawAudio,
#             generator_loss=mel_gan_gen_loss,
#             discriminator_loss=mel_gan_disc_loss)
#
#
# class LargeReceptiveFieldFilterBankExperiment(Experiment):
#     """
#     Surprisingly, this experiment with larger receptive fields in the
#     discriminator ends up with much *less* coherence, with jittery, short
#     frames of realistic speech that form no coherent larger-scale picture.
#     """
#
#     def __init__(self):
#         feature_size = 64
#         sr = zounds.SR11025()
#         scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), 128)
#         filter_bank = zounds.learn.FilterBank(
#             sr, 511, scale, 0.9, normalize_filters=True, a_weighting=False)
#         super().__init__(
#             generator=FilterBankGenerator(filter_bank),
#             discriminator=LargeReceptiveFieldFilterBankDiscriminator(
#                 filter_bank),
#             learning_rate=1e-4,
#             feature_size=feature_size,
#             audio_repr_class=RawAudio,
#             generator_loss=mel_gan_gen_loss,
#             discriminator_loss=mel_gan_disc_loss)


class FilterBankExperiment(Experiment):
    """
    This is probably the best audio quality yet.  The audio is relatively
    crisp, spectrograms are indistinguishable from real speech, although they
    are hard to understand.

    There are definite phase issues here and there after 12 hours.

    Overall, the texture of the speech is more realistic than what's produced
    by the basic MelGAN setup.
    """

    def __init__(self):
        n_mels = 128
        feature_size = 32
        sr = zounds.SR22050()
        n_fft = 1024
        hop = 256
        total_samples = 8192

        scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), 128)
        filter_bank = zounds.learn.FilterBank(
            sr, 511, scale, 0.9, normalize_filters=True, a_weighting=False)

        spec_func = make_spectrogram_func(
            normalized_and_augmented_audio, sr, n_fft, hop, n_mels)

        super().__init__(
            generator=FilterBankGenerator(
                filter_bank, feature_size, total_samples, n_mels),
            discriminator=FilterBankDiscriminator(filter_bank, total_samples),
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
