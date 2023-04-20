import os
import glob
from pydub import AudioSegment

def find_wav_files(directory):
    wav_files = []
    for file in glob.glob(os.path.join(directory, '*.wav')):
        wav_files.append(file)
    return wav_files

def get_smallest_duration(wav_files):
    smallest_duration = float('inf')
    for file in wav_files:
        audio = AudioSegment.from_wav(file)
        duration = len(audio)
        if duration < smallest_duration:
            smallest_duration = duration
    return smallest_duration

def split_audio_files(wav_files, smallest_duration, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    chunk_counter = 1
    for file in wav_files:
        audio = AudioSegment.from_wav(file)
        for i in range(0, len(audio), int(smallest_duration/5)):
            end_index = i + smallest_duration
            if end_index > len(audio):
                end_index = len(audio)
                start_index = end_index - smallest_duration
            else:
                start_index = i

            chunk = audio[start_index:end_index]
            chunk.export(os.path.join(output_directory, f'{chunk_counter}.wav'), format='wav')
            chunk_counter += 1

def main():
    input_directory = '../20samples'
    output_directory = '../raw5'

    wav_files = find_wav_files(input_directory)
    smallest_duration = get_smallest_duration(wav_files)
    split_audio_files(wav_files, smallest_duration, output_directory)

if __name__ == '__main__':
    main()
