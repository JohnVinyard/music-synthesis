from ..audio import RawAudio
from ..train import GeneratorTrainer, DiscriminatorTrainer
from ..experiment import FilterBankExperiment
from ..generator import FilterBankGenerator
from ..featuregenerator import LowResGenerator
from ..featurediscriminator import LowResDiscriminator
from .init import weights_init
from ..data import batch_stream
from ..feature import normalized_and_augmented_audio, make_spectrogram_func
from torch.optim import Adam
import torch
from itertools import cycle
import zounds

class BaseVocoder(object):
    def __call__(self, features):
        raise NotImplementedError()


class PhaseRecoveryVocoder(BaseVocoder):
    # TODO: Use a the BasePhaseRecoveryVocoder from the audio representation
    # module
    pass

class NeuralVocoder(object):
    def __init__(self, network, weights_path):
        if not weights_path:
            raise ValueError('You must supply a path for pre-trained weights')
        self.network = network
        self.network.load_state_dict(torch.load(weights_path))

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
            audio_repr_class,
            learning_rate,
            condition_shape,
            samplerate):

        super().__init__()

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
            self.d_trainer,
            self.g_trainer
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
        batch_size = len(batch)
        conditioning = torch.normal(
            0, 1, (batch_size,) + self.condition_shape).to(self.device)
        return batch, conditioning

    def features_to_audio(self, features):
        # TODO: Use the audio representation class here, instead of naively
        # converting directly to raw audio
        features = torch.from_numpy(features).to(self.device)
        audio = self.vocoder(features)
        output = [
            zounds.AudioSamples(a, self.samplerate)
            for a in audio.data.cpu().numpy().squeeze()]
        return output




class TestFeatureExperiment(BaseFeatureExperiment):
    pass


class TwoDimGeneratorFeatureExperiment(BaseFeatureExperiment):



    def __init__(self):
        # samplerate = zounds.SR22050()
        # # TODO: Get these values from the FilterBankExperiment
        # generator_size = 32
        # generator_samples = 8192
        # n_mels = 128
        # n_fft = 1024
        # hop = 256
        # # END TODO: get these values from the FilterBankExperiment
        #
        # # TODO: I actually need to use the conditional filter bank experiment
        # # here
        # generator = FilterBankGenerator(
        #     FilterBankExperiment.make_filter_bank(samplerate),
        #     generator_size,
        #     generator_samples,
        #     n_mels)
        #
        # spec_func = make_spectrogram_func(
        #     normalized_and_augmented_audio, samplerate, n_fft, hop, n_mels)

        # Parameters for this experiment
        noise_dim = 128


        # TODO: I actually need to use the conditional filter bank experiment
        # here. All experiments, or at least those I care about using in this
        # second phase, should implement this basic class-level interface
        vocoder_exp = FilterBankExperiment
        generator = vocoder_exp.make_generator()
        spec_func = vocoder_exp.make_spec_func()
        samplerate = vocoder_exp.SAMPLERATE

        disc_channels = 256

        # TODO: create generator and discriminator loss functions and consider
        # how they'll function in the current trainers
        super().__init__(
            vocoder=NeuralVocoder(generator),
            feature_generator=LowResGenerator(
                out_channels=vocoder_exp.N_MELS,
                noise_dim=noise_dim),
            generator_init=weights_init,
            generator_loss=None,
            feature_disc=LowResDiscriminator(
                feature_channels=vocoder_exp.N_MELS,
                channels=disc_channels),
            disc_init=weights_init,
            disc_loss=None,
            feature_funcs={
                'spectrogram': (spec_func, (samplerate,))
            },
            audio_repr_class=RawAudio,
            learning_rate=1e-4,
            condition_shape=(noise_dim, 1),
            samplerate=samplerate)