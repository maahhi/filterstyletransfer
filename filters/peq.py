import numpy as np
import soundfile as sf
from scipy.signal import lfilter, butter

def equalizer(wav_file, eq_gains ,output_file="", save = False):
    eq_bands = [
        {'low_freq': 20, 'high_freq': 100, 'gain': eq_gains[0]},
        {'low_freq': 100, 'high_freq': 200, 'gain': eq_gains[1]},
        {'low_freq': 200, 'high_freq': 300, 'gain': eq_gains[2]},
        {'low_freq': 300, 'high_freq': 400, 'gain': eq_gains[3]},
        {'low_freq': 400, 'high_freq': 500, 'gain': eq_gains[4]},
        {'low_freq': 500, 'high_freq': 12000, 'gain': eq_gains[5]}
    ]
    '''
    eq_bands = [
        {'low_freq': 20, 'high_freq': 100, 'gain': eq_gains[0]},
        {'low_freq': 100, 'high_freq': 500, 'gain': eq_gains[1]},
        {'low_freq': 500, 'high_freq': 2000, 'gain': eq_gains[2]},
        {'low_freq': 2000, 'high_freq': 4000, 'gain': eq_gains[3]},
        {'low_freq': 4000, 'high_freq': 8000, 'gain': eq_gains[4]},
        {'low_freq': 8000, 'high_freq': 12000, 'gain': eq_gains[5]}
    ]'''

    # Load the WAV file
    data, samplerate = sf.read(wav_file)

    # Initialize the processed data
    processed_data = np.zeros_like(data)

    # Process the audio data with an equalizer
    for band in eq_bands:
        low_freq = band['low_freq'] / (samplerate / 2)
        high_freq = min(band['high_freq'] / (samplerate / 2), 0.99) # Ensure Wn < 1

        if low_freq >= high_freq:
            raise ValueError("Low frequency must be less than high frequency")

        b, a = butter(1, [low_freq, high_freq], btype='band')
        band_data = lfilter(b, a, data)
        band_data *= 10 ** (band['gain'] / 20)
        processed_data += band_data

    # Save the processed data to a new WAV file
    if save:
        sf.write(output_file, processed_data, samplerate)
    return processed_data

# Example usage:
#eq_gains = [20,20,20,-20,-20,-20]

# gain can be between -20 to +20


#equalizer("inpu.wav", eq_gains)

