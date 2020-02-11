from ..train import GeneratorTrainer, DiscriminatorTrainer
from ..feature import total_samples, feature_channels
from torch.optim import Adam
import torch


class BaseGanExperiment(object):
    def _init__(self):
        super().__init__()

    @property
    def generator(self):
        raise NotImplementedError()

    @property
    def discriminator(self):
        raise NotImplementedError()

    @property
    def generator_trainer(self):
        raise NotImplementedError()

    @property
    def discriminator_trainer(self):
        raise NotImplementedError()

    @property
    def feature_spec(self):
        raise NotImplementedError()

    def from_audio(self, samples, sr):
        raise NotImplementedError()

    def audio_representation(self, data, sr):
        raise NotImplementedError()

    def to(self, device):
        self.generator.to(device)
        self.discriminator.to(device)
        return self


class Experiment(BaseGanExperiment):
    def __init__(
            self,
            generator,
            discriminator,
            learning_rate,
            feature_size,
            audio_repr_class,
            generator_loss,
            discriminator_loss):
        super().__init__()

        self.__g = generator.initialize_weights()
        self.__g_optim = Adam(
            self.__g.parameters(), lr=learning_rate, betas=(0, 0.9))

        self.__d = discriminator.initialize_weights()
        self.__d_optim = Adam(
            self.__d.parameters(), lr=learning_rate, betas=(0, 0.9))

        self.__g_trainer = GeneratorTrainer(
            self.__g,
            self.__g_optim,
            self.__d,
            self.__d_optim,
            generator_loss)

        self.__d_trainer = DiscriminatorTrainer(
            self.__g,
            self.__g_optim,
            self.__d,
            self.__d_optim,
            discriminator_loss)

        self.__feature_size = feature_size
        self.__audio_repr_class = audio_repr_class

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
        torch.save(self.generator.state_dict(), self._gen_name())
        torch.save(self.discriminator.state_dict(), self._disc_name())

    def resume(self):
        self.generator.load_state_dict(torch.load(self._gen_name()))
        self.discriminator.load_state_dict(torch.load(self._disc_name()))

    @property
    def generator(self):
        return self.__g

    @property
    def discriminator(self):
        return self.__d

    @property
    def generator_trainer(self):
        return self.__g_trainer.train

    @property
    def discriminator_trainer(self):
        return self.__d_trainer.train

    def from_audio(self, samples, sr):
        return self.__audio_repr_class.from_audio(samples, sr)

    def audio_representation(self, data, sr):
        return self.__audio_repr_class(data, sr)

    @property
    def feature_spec(self):
        return {
            'audio': (total_samples, 1),
            'spectrogram': (self.__feature_size, feature_channels)
        }
