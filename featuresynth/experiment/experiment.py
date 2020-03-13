from ..train import GeneratorTrainer, DiscriminatorTrainer
from .init import weights_init
from ..data import batch_stream
from ..loss import hinge_discriminator_loss, hinge_generator_loss
from torch.optim import Adam
import torch
from itertools import cycle
import zounds


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
            g_init=weights_init,
            d_init=weights_init,
            feature_funcs=None,
            total_samples=16384,
            feature_channels=256,
            inference_sequence_factor=4,
            samplerate=zounds.SR11025(),
            sub_disc_loss=hinge_discriminator_loss,
            sub_gen_loss=hinge_generator_loss):

        super().__init__()

        # how much longer than the test sequence should the inference sequence
        # be?
        self.sub_gen_loss = sub_gen_loss
        self.sub_disc_loss = sub_disc_loss
        self.inference_sequence_factor = inference_sequence_factor
        if feature_funcs is None:
            raise ValueError('You must provide feature funcs')

        self.discriminator_init = d_init
        self.generator_init = g_init

        if hasattr(generator, 'initialize_weights'):
            raise ValueError(
                'initialize_weights() method on generators is deprecated')

        if hasattr(discriminator, 'initialize_weights'):
            raise ValueError(
                'initialize_weights() method on discriminators is deprecated')

        self.__g = generator
        self.__g.apply(g_init)
        self.__g_optim = Adam(
            self.__g.parameters(), lr=learning_rate, betas=(0.5, 0.9))

        self.__d = discriminator
        self.__d.apply(d_init)
        self.__d_optim = Adam(
            self.__d.parameters(), lr=learning_rate, betas=(0.5, 0.9))

        self.__g_trainer = GeneratorTrainer(
            self.__g,
            self.__g_optim,
            self.__d,
            self.__d_optim,
            generator_loss,
            self.sub_gen_loss)

        self.__d_trainer = DiscriminatorTrainer(
            self.__g,
            self.__g_optim,
            self.__d,
            self.__d_optim,
            discriminator_loss,
            self.sub_disc_loss)

        self.__feature_size = feature_size
        self.__audio_repr_class = audio_repr_class

        self.__anchor_feature = 'spectrogram'
        self.__feature_funcs = feature_funcs

        self.training_steps = cycle([
            self.discriminator_trainer,
            self.generator_trainer
        ])

        self.samplerate = samplerate
        self.total_samples = total_samples
        self.feature_channels = feature_channels

    @classmethod
    def _name(cls):
        name = cls.__name__.lower()
        name = name.replace('experiment', '')
        return name

    @classmethod
    def _gen_name(cls, prefix=''):
        name = cls._name()
        return f'{prefix}{name}_gen.dat'

    @classmethod
    def _disc_name(cls, prefix=''):
        name = cls._name()
        return f'{prefix}{name}_disc.dat'

    @classmethod
    def load_generator_weights(cls, generator, prefix=''):
        generator.load_state_dict(torch.load(cls._gen_name(prefix)))
        return generator

    def checkpoint(self, prefix=''):
        torch.save(self.generator.state_dict(), self._gen_name(prefix))
        torch.save(self.discriminator.state_dict(), self._disc_name(prefix))

    def resume(self, prefix=''):
        self.generator.load_state_dict(torch.load(self._gen_name(prefix)))
        self.discriminator.load_state_dict(torch.load(self._disc_name(prefix)))

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

    def preprocess_batch(self, batch):
        samples, features = batch
        r = self.from_audio(samples, self.samplerate)
        samples = r.data
        return samples, features

    def batch_stream(self, path, pattern, batch_size, feature_spec=None):
        return batch_stream(
            path,
            pattern,
            batch_size,
            feature_spec or self.feature_spec,
            self.__anchor_feature,
            self.__feature_funcs)

    @property
    def feature_spec(self):
        return {
            'audio': (self.total_samples, 1),
            'spectrogram': (self.__feature_size, self.feature_channels)
        }

    @property
    def inference_spec(self):
        inf = {}
        for k, v in self.feature_spec.items():
            size, channels = v
            inf[k] = (size * self.inference_sequence_factor, channels)
        return inf
