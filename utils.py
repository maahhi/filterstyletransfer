import os
import random


def load_wav_files(path):
    wav_files = []
    for file in os.listdir(path):
        if file.endswith(".wav"):
            wav_files.append(os.path.join(path, file))
    return wav_files

