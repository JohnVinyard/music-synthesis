
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

    @property
    def overfit(self):
        return False

    def to(self, device):
        self.generator.to(device)
        self.discriminator.to(device)
        return self


