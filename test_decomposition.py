from featuresynth.audio.transform import \
    fft_frequency_decompose, fft_frequency_recompose
from featuresynth.data import iter_files
import zounds
import torch

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(port=9999)

    file_path = next(iter_files('/hdd/LJSpeech-1.1', '*.wav'))
    samples = zounds.AudioSamples.from_file(file_path).mono

    short = samples[:8192]
    long = samples[:32768]

    short_bands = fft_frequency_decompose(
        torch.from_numpy(short)[None, None, :], 512)
    long_bands = fft_frequency_decompose(
        torch.from_numpy(long)[None, None, :], 2048)


    short_recon = fft_frequency_recompose(short_bands, 8192).data.cpu().numpy().squeeze()
    long_recon = fft_frequency_recompose(long_bands, 32768).data.cpu().numpy().squeeze()

    short_recon = zounds.AudioSamples(short_recon, samples.samplerate)
    long_recon = zounds.AudioSamples(long_recon, samples.samplerate)

    input('waiting')