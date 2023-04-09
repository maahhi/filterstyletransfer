import numpy as np
import soundfile as sf
from scipy.io import wavfile

def distortion(wav_file, output_file, gain, threshold):
    # Load the WAV file
    samplerate, data = wavfile.read(wav_file)
    data = data.astype(np.float32) / 32768.0  # Convert to float

    # Apply distortion
    data = data * gain

    # Clip the audio data based on the threshold
    data[data > threshold] = threshold
    data[data < -threshold] = -threshold

    # Normalize the audio data
    data = data / np.max(np.abs(data))

    # Convert back to int16
    processed_data = (data * 32767).astype(np.int16)

    # Save the processed data to a new WAV file
    wavfile.write(output_file, samplerate, processed_data)

# Example usage:
gain = 10
threshold = 0.4
distortion("mug1.wav", "output_distorted.wav", gain, threshold)
