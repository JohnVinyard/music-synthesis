from ..train import GeneratorTrainer, DiscriminatorTrainer
from ..feature import total_samples, feature_channels
from .init import generator_init, discriminator_init
from torch.optim import Adam
import torch
import numpy as np


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

    def preprocess_batch(self, batch):
        return batch

    def to(self, device):
        self.generator.to(device)
        self.discriminator.to(device)
        return self

    def report(self, n):
        """
        Produce an HTML report displaying N examples that each include:
            - real audio
            - real stft
            - conditioning feature
            - generated audio
            - generated stft
            - report on GPU and CPU generation times

        These files are saved to a local directory, or uploaded to an s3
        bucket for public access

        They are placed into a simple HTML template that includes vue.js,
        which renders the results
        """
        raise NotImplementedError()


class Experiment(BaseGanExperiment):
    def __init__(
            self,
            generator,
            discriminator,
            learning_rate,
            feature_size,
            audio_repr_class,
            generator_loss,
            discriminator_loss,
            g_init=generator_init,
            d_init=discriminator_init):
        super().__init__()

        self.discriminator_init = d_init
        self.generator_init = g_init

        if hasattr(generator, 'initialize_weights'):
            raise ValueError(
                'initialize_weights() method on generators is deprecated')

        if hasattr(discriminator, 'initialize_weights'):
            raise ValueError(
                'initialize_weights() method on discriminators is deprecated')

        self.__g = generator
        self._apply_init(self.__g, self.generator_init)
        self.__g_optim = Adam(
            self.__g.parameters(), lr=learning_rate, betas=(0, 0.9))

        self.__d = discriminator
        self._apply_init(self.__d, self.discriminator_init)
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

    def _apply_init(self, network, init_func):
        for name, weight in network.named_parameters():
            init_func(name, weight)

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

    def batch_stream(self, batch_size, data_source, anchor_feature):
        return data_source.batch_stream(
            batch_size, self.feature_spec, anchor_feature)

    def preprocess_batch(self, batch):
        samples, features = batch

        # max one normalization for samples
        samples /= np.abs(samples).max(axis=-1, keepdims=True) + 1e-12

        # max one normalization for features, which may have had log scaling
        # applied and might be negative
        features -= features.min(axis=(1, 2), keepdims=True)
        features /= features.max(axis=(1, 2), keepdims=True) + 1e-12
        return samples, features

    @property
    def feature_spec(self):
        return {
            'audio': (total_samples, 1),
            'spectrogram': (self.__feature_size, feature_channels)
        }
