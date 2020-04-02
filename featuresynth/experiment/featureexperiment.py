from ..audio import RawAudio
from ..audio.representation import BasePhaseRecovery
from ..train import GeneratorTrainer, DiscriminatorTrainer
from ..experiment import FilterBankMultiscaleExperiment
from ..featuregenerator import SpectrogramFeatureGenerator
from ..featurediscriminator import SpectrogramFeatureDiscriminator
from ..feature import spectrogram
from .init import weights_init
from ..data import batch_stream
from ..loss import least_squares_generator_loss, least_squares_disc_loss
from torch.optim import Adam
import torch
from itertools import cycle
import numpy as np
from librosa.filters import mel

def make_phase_vocoder_class(samplerate, n_fft, n_mels):

    class PhaseRecovery(BasePhaseRecovery):
        basis = mel(
            sr=int(samplerate),
            n_fft=n_fft,
            n_mels=n_mels)

    return PhaseRecovery

class BaseVocoder(object):
    def __call__(self, features):
        raise NotImplementedError()

class DeterministicVocoder(BaseVocoder):
    def __init__(self, repr_class, samplerate):
        super().__init__()
        self.samplerate = samplerate
        self.repr_class = repr_class

    def __call__(self, features):
        try:
            features = features.data.cpu().numpy()
        except AttributeError:
            pass
        r = self.repr_class(features, self.samplerate)
        return r.to_audio()

class NeuralVocoder(BaseVocoder):
    def __init__(self, network):
        self.network = network

    def __call__(self, features):
        with torch.no_grad():
            return self.network(features)


# TODO: Much of this implementation is based on or copied from
# BaseGanExperiment and Experiment and could use some refactoring
class BaseFeatureExperiment(object):
    """
    Derived classes implement an experiment that trains a generator to create
    novel features, probably spectograms, which can be fed to a pre-trained
    vocoder.

    Ideally, this experiment should be flexible enough to support both GAN
    and autoregressive settings

    The discriminator here might be a GAN discriminator, or a module that
    computes an autoregressive loss and has no parameters
    """

    def __init__(
            self,
            vocoder,
            feature_generator,
            generator_init,
            generator_loss,
            feature_disc,
            disc_init,
            disc_loss,
            feature_funcs,
            feature_spec,
            audio_repr_class,
            learning_rate,
            condition_shape,
            samplerate):

        super().__init__()

        self.feature_spec = feature_spec
        self.samplerate = samplerate
        self.condition_shape = condition_shape
        self.disc_loss = disc_loss
        self.generator_loss = generator_loss
        self.disc_init = disc_init
        self.generator_init = generator_init
        self.learning_rate = learning_rate
        self.feature_disc = feature_disc
        self.audio_repr_class = audio_repr_class
        self.feature_funcs = feature_funcs
        self.feature_generator = feature_generator
        # a vocoder is needed, regardless of whether this is a GAN or
        # autoregressive setting
        self.vocoder = vocoder
        self.device = None

        self.feature_generator.apply(self.generator_init)
        self.feature_disc.apply(self.disc_init)

        self.g_optim = Adam(
            self.feature_generator.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.9))

        self.d_optim = Adam(
            self.feature_disc.parameters(),
            lr=learning_rate,
            betas=(0.5, 0.9))

        self.g_trainer = GeneratorTrainer(
            self.feature_generator,
            self.g_optim,
            self.feature_disc,
            self.d_optim,
            self.generator_loss)

        self.d_trainer = DiscriminatorTrainer(
            self.feature_generator,
            self.g_optim,
            self.feature_disc,
            self.d_optim,
            self.disc_loss)


        self.training_steps = cycle([
            self.d_trainer.train,
            self.g_trainer.train
        ])

    def to(self, device):
        self.feature_generator.to(device)
        self.feature_disc.to(device)
        self.device = device
        return self

    # TODO: All of the following (through checkpoint() and resume()) is copied
    # almost verbatim from Experiment and should probably be factored into
    # a common location
    def _name(self):
        name = self.__class__.__name__.lower()
        name = name.replace('experiment', '')
        return name

    def _gen_name(self):
        name = self._name()
        return f'{name}_gen.dat'

    def _disc_name(self):
        name = self._name()
        return f'{name}_disc.dat'

    def checkpoint(self):
        torch.save(self.feature_generator.state_dict(), self._gen_name())
        torch.save(self.feature_disc.state_dict(), self._disc_name())

    def resume(self):
        self.feature_generator.load_state_dict(torch.load(self._gen_name()))
        self.feature_disc.load_state_dict(torch.load(self._disc_name()))

    def batch_stream(self, path, pattern, batch_size):
        return batch_stream(
            path,
            pattern,
            batch_size,
            self.feature_spec,
            'spectrogram',
            self.feature_funcs)

    def preprocess_batch(self, batch):
        """
        Preprocess a batch of real spectrograms
        Also, produce a conditioning vector
        """
        # unpack single-item tuple
        spec, = batch
        batch_size = len(spec)
        conditioning = np.random.normal(
            0, 1, (batch_size,) + self.condition_shape)
        return spec, conditioning

    def features_to_audio(self, features):
        batch_size = features.shape[0]
        features = torch.from_numpy(features)

        audio_features = self.vocoder(features)

        try:
            audio_features = audio_features.data.cpu().numpy()
        except AttributeError:
            audio_features = \
                {k:v.data.cpu().numpy() for k, v in audio_features.items()}

        # audio_features = audio_features.reshape((batch_size, 1, -1))
        return self.audio_repr_class(audio_features, self.samplerate)


class TestFeatureExperiment(BaseFeatureExperiment):
    pass


class TwoDimGeneratorFeatureExperiment(BaseFeatureExperiment):

    def __init__(self):
        noise_dim = 128


        vocoder_exp = FilterBankMultiscaleExperiment
        samplerate = vocoder_exp.SAMPLERATE
        generator = vocoder_exp.make_generator()
        generator = vocoder_exp.load_generator_weights(generator)
        vocoder = NeuralVocoder(generator)

        disc_channels = 256

        def gen_loss(r_features, f_features, r_score, f_score, gan_loss):
            return least_squares_generator_loss(f_score)

        def disc_loss(r_score, f_score, gan_loss):
            return least_squares_disc_loss(r_score, f_score)


        super().__init__(
            vocoder=vocoder,
            feature_generator=SpectrogramFeatureGenerator(
                out_channels=vocoder_exp.N_MELS,
                noise_dim=noise_dim),
            generator_init=weights_init,
            generator_loss=gen_loss,
            feature_disc=SpectrogramFeatureDiscriminator(
                feature_channels=vocoder_exp.N_MELS,
                channels=disc_channels),
            disc_init=weights_init,
            disc_loss=disc_loss,
            feature_funcs={
                'spectrogram': (spectrogram, (samplerate,))
            },
            feature_spec={
                'spectrogram': (512, vocoder_exp.N_MELS)
            },
            audio_repr_class=vocoder_exp.AUDIO_REPR_CLASS,
            learning_rate=1e-4,
            condition_shape=(noise_dim, 1),
            samplerate=samplerate)