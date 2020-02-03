from fnmatch import fnmatch
import os


def iter_files(base_path, pattern):
    for dirpath, dirnames, filenames in os.walk(base_path):
        audio_files = filter(
            lambda x: fnmatch(x, pattern),
            (os.path.join(dirpath, fn) for fn in filenames))
        yield from audio_files
