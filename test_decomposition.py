from featuresynth.audio.transform import \
    fft_frequency_decompose, fft_frequency_recompose
from featuresynth.audio import ComplextSTFT, MultiScale
from featuresynth.data import iter_files
import zounds
import torch


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(port=9999)

    file_path = next(iter_files('/hdd/LJSpeech-1.1', '*.wav'))
    speech = zounds.AudioSamples.from_file(file_path).mono[:32768]
    r = MultiScale.from_audio(speech[None, None, :], speech.samplerate)
    recon = r.listen()

    input('waiting')
