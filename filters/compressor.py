import numpy as np
import soundfile as sf
from scipy.signal import lfilter
from pydub import AudioSegment

def compressor(data, samplerate, threshold, ratio, attack=5, release=100):

    # Calculate the linear threshold value
    threshold_linear = 10 ** (threshold / 20)

    # Convert attack and release times to samples
    attack_samples = int(attack * samplerate / 1000)
    release_samples = int(release * samplerate / 1000)

    # Initialize gain and the smoothed gain array
    gain = 1.0
    smoothed_gain = np.ones_like(data)

    # Process the audio
    for i in range(1, len(data)):
        # Calculate the ideal gain for the current sample
        if data[i] > threshold_linear:
            gain = threshold_linear / data[i]
            gain = max(min(gain, 1.0), 1.0 / ratio)

        # Smooth the gain using attack and release times
        if gain < smoothed_gain[i - 1]:
            coeff = 1.0 / attack_samples
        else:
            coeff = 1.0 / release_samples

        smoothed_gain[i] = coeff * gain + (1.0 - coeff) * smoothed_gain[i - 1]

    # Apply the gain to the audio data
    return data * smoothed_gain

def apply_compressor(input_file, output_file, threshold, ratio, attack, release):
    # Load the WAV file
    audio_segment = AudioSegment.from_wav(input_file)
    data = np.array(audio_segment.get_array_of_samples(), dtype=np.float64) / (2 ** 15)
    samplerate = audio_segment.frame_rate

    # Apply the compressor
    compressed_data = compressor(data, samplerate, threshold, ratio, attack, release)

    # Normalize the audio
    compressed_data = (compressed_data / np.max(np.abs(compressed_data))) * (2 ** 15 - 1)

    # Save the compressed audio to a new WAV file
    sf.write(output_file, compressed_data.astype(np.int16), samplerate)



if __name__ == "__main__":
    # Example usage:
    input_file = "1.wav"
    output_file = "1compressed.wav"
    threshold = -20  # dB (-20,20)
    ratio = 4        # 4:1 compression ratio (2,6)

    apply_compressor(input_file, output_file, threshold, ratio)

