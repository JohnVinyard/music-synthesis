from torch.optim import Adam

from .experiment import BaseGanExperiment
from ..audio import MDCT
from ..discriminator import MDCTDiscriminator
from ..feature import total_samples, feature_channels
from ..generator import MDCTGenerator
from ..train import GeneratorTrainer, DiscriminatorTrainer
from ..util.modules import least_squares_disc_loss, least_squares_generator_loss


class MDCTExperiment(BaseGanExperiment):
    def __init__(self):
        super().__init__()

        learning_rate = 1e-4

        self.__g = MDCTGenerator(feature_channels).initialize_weights()
        self.__g_optim = Adam(
            self.__g.parameters(), lr=learning_rate, betas=(0, 0.9))

        self.__d = MDCTDiscriminator(feature_channels).initialize_weights()
        self.__d_optim = Adam(
            self.__d.parameters(), lr=learning_rate, betas=(0, 0.9))

        self.__g_trainer = GeneratorTrainer(
            self.__g,
            self.__g_optim,
            self.__d,
            self.__d_optim,
            least_squares_generator_loss)

        self.__d_trainer = DiscriminatorTrainer(
            self.__g,
            self.__g_optim,
            self.__d,
            self.__d_optim,
            least_squares_disc_loss)

        self.__feature_size = 64

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
        return MDCT.from_audio(samples, sr)

    def audio_representation(self, data, sr):
        return MDCT(data, sr)

    @property
    def feature_spec(self):
        return {
            'audio': (total_samples, 1),
            'spectrogram': (self.__feature_size, feature_channels)
        }
