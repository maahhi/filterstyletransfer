import os
import glob
from pydub import AudioSegment

def print_wav_durations(directory):
    for file in glob.glob(os.path.join(directory, '*.wav')):
        audio = AudioSegment.from_wav(file)
        duration = len(audio) / 1000  # Convert to seconds
        print(f"File: {file}, Duration: {duration:.2f} seconds")

input_directory = '../raw'
print_wav_durations(input_directory)
