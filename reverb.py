import numpy as np
import soundfile as sf
from scipy.signal import convolve

def add_reverb(wav_file, output_file, room_scale=0.8, wet_level=0.5, dry_level=1.0):
    # Load the WAV file
    data, samplerate = sf.read(wav_file)

    # Generate an impulse response based on the room_scale
    impulse_response = np.zeros(int(samplerate * room_scale))
    for i in range(len(impulse_response)):
        impulse_response[i] = (room_scale - i / samplerate) / room_scale
        impulse_response[i] *= np.random.uniform(-1, 1)

    # Convolve the input audio data with the impulse response
    reverb_data = convolve(data, impulse_response)

    # Normalize the reverb data
    reverb_data /= np.max(np.abs(reverb_data))

    # Mix the dry and wet signals
    reverb_length = len(reverb_data) - len(data)
    padded_data = np.pad(data, (0, reverb_length), mode='constant')
    processed_data = dry_level * padded_data + wet_level * reverb_data

    # Save the processed data to a new WAV file
    sf.write(output_file, processed_data, samplerate)

# Example usage:
add_reverb("mug1.wav", "output_reverb.wav", room_scale=0.8, wet_level=0.5, dry_level=1.0)
