import numpy as np
# import torch
# import zounds
import librosa
# from featuresynth.data import DataStore
# from featuresynth.feature.spectrogram import FilterBankSpectrogram
# from featuresynth.audio import MelScalePhaseRecover, GeometricScalePhaseRecover
import time
from featuresynth.data.conjure import cache, LmdbCollection
from featuresynth.data.filesystem import iter_files
from featuresynth.audio import ComplextSTFT
import zounds
import torch



if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    sr = zounds.SR22050()
    # synth = zounds.SineSynthesizer(sr)
    # samples = synth.synthesize(
    #     zounds.Seconds(2), [110, 220, 440, 880]).astype(np.float32)
    file_path = next(iter_files('/hdd/LJSpeech-1.1', '*.wav'))
    samples = zounds.AudioSamples.from_file(file_path).astype(np.float32)

    r = ComplextSTFT.from_audio(samples[None, None, :], sr)
    phase = r.phase
    phase[:] = np.random.uniform(-np.pi, np.pi, phase.shape)
    recon = r.listen()

    scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist - 20), 256)
    filter_bank = zounds.learn.FilterBank(
        sr, 1024, scale, 0.5, normalize_filters=False, a_weighting=False)

    result = filter_bank.convolve(torch.from_numpy(samples)[None, :])
    spec = np.clip(result.data.cpu().numpy().squeeze(), 0, np.inf).T[1024: 2048]

    phase_result = filter_bank.convolve(torch.from_numpy(recon)[None, :])
    phase_spec = np.clip(phase_result.data.cpu().numpy().squeeze(), 0, np.inf).T[1024:2048]
    input('Waiting...')

    # path = '/hdd/TIMIT'
    # full_path = next(iter_files(path, '*.WAV'))


    # collection = LmdbCollection('idata')
    #
    # def make_resampler(samplerate):
    #
    #     @cache(collection)
    #     def f(x):
    #         return librosa.resample(x, int(x.samplerate), int(samplerate))
    #
    #     return f






    # from featuresynth.experiment import Report
    # from featuresynth.experiment.winners import MultiScaleMelGanExperiment
    # experiment = MultiScaleMelGanExperiment()
    # r = Report(experiment, 'test-generator-report')
    # r.generate(ds, 3, sr, regenerate=True)

    # total_examples = 0
    # for feature in ds.iter_feature('spectrogram', 256):
    #     frames, channels = feature.shape
    #     total_examples += (frames - 64) + 1
    #     print(total_examples)


    # samples = zounds.AudioSamples.from_file(
    #     '/hdd/musicnet/train_data/2315.wav')
    #
    # start = time.time()
    # rs1 = zounds.soundfile.resample(samples, zounds.SR11025())
    # print(f'took {time.time() - start} seconds')
    #
    # start = time.time()
    # rs2 = librosa.resample(samples, int(samples.samplerate), 11025)
    # print(f'took {time.time() - start} seconds')