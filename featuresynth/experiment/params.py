import zounds


class ExperimentParameters(object):
    def __init__(
            self,
            samplerate,
            feature_hop,
            feature_window,
            training_sample_win,
            downsampling_ratios):
        super().__init__()
        self.downsampling_ratios = downsampling_ratios
        self.training_sample_win = training_sample_win
        self.feature_window = feature_window
        self.feature_hop = feature_hop
        self.samplerate = samplerate

    @property
    def feature_hop_hz(self):
        return \
            zounds.Seconds(1) / (self.samplerate.frequency * self.feature_hop)

    @property
    def feature_window_len(self):
        return (self.samplerate.frequency * self.feature_window) / zounds.Seconds(1)

    @property
    def training_sample_len(self):
        return (self.samplerate.frequency * self.training_sample_win) / zounds.Seconds(1)

    @property
    def feature_size(self):
        return self.training_sample_win // self.feature_hop

    @property
    def upsampling_ratio(self):
        return self.training_sample_win // self.feature_size

    @property
    def judgement_hz(self):
        return [zounds.Seconds(1) / (self.samplerate.frequency * dsr) for dsr in self.downsampling_ratios]

    def report(self):
        print('==============================')
        print(f'Feature samplerate: {self.feature_hop_hz} hz')
        print(f'Feature window len: {self.feature_window_len} seconds')
        print(f'Training sample len: {self.training_sample_len} seconds')
        print(f'Feature dim: {self.feature_size} frames')
        print(f'Upsample ratio: {self.upsampling_ratio}')
        print(f'Judgement hz: {self.judgement_hz}')


if __name__ == '__main__':
    mel_gan = ExperimentParameters(
        samplerate=zounds.SR22050(),
        feature_hop=256,
        feature_window=1024,
        training_sample_win=8192,
        downsampling_ratios=[256, 512, 1024])
    mel_gan.report()

    mine = ExperimentParameters(
        samplerate=zounds.SR11025(),
        feature_hop=256,
        feature_window=1024,
        training_sample_win=16384,
        downsampling_ratios=[128, 256, 512])
    mine.report()
