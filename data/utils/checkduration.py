import os
import glob
from pydub import AudioSegment

def print_wav_durations(directory):
    for file in glob.glob(os.path.join(directory, '*.wav')):
        audio = AudioSegment.from_wav(file)
        duration = len(audio) / 1000  # Convert to seconds
        sample_rate = audio.frame_rate
        print(f"File: {file}, Duration: {duration:.2f} seconds",sample_rate)

input_directory = '../raw'
print_wav_durations(input_directory)
