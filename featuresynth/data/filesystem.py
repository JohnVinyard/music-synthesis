from fnmatch import fnmatch
import os
import soundfile

def iter_files(base_path, pattern):
    for dirpath, dirnames, filenames in os.walk(base_path):
        audio_files = filter(
            lambda x: fnmatch(x, pattern),
            (os.path.join(dirpath, fn) for fn in filenames))
        yield from audio_files


def iter_audio_chunks(base_path, pattern, chunk_size_seconds=30):
    for filepath in iter_files(base_path, pattern):
        info = soundfile.info(filepath)
        step = info.samplerate * chunk_size_seconds
        for i in range(0, info.frames, step):
            start = i
            stop = min(info.frames, start + step)
            yield filepath, start, stop
