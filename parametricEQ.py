import numpy as np
from scipy.signal import biquad, iirpeak
from pydub import AudioSegment

def apply_parametric_eq(wav_file, eq_params):
    audio = AudioSegment.from_wav(wav_file)
    samples = np.array(audio.get_array_of_samples(), dtype=float)
    fs = audio.frame_rate

    for fc, q, gain in eq_params:
        b, a = iirpeak(fc, q, gain, fs)
        samples = biquad(samples, b, a)

    eq_audio = audio._spawn(np.array(samples, dtype=np.int16).tobytes())
    return eq_audio

# Example usage:
eq_params = [
    (100, 1.5, 6),    # (fc, q, gain) in Hz, Q factor, dB
    (500, 1.2, -3),
    (1000, 2, 4),
    (2500, 1.8, -6),
    (5000, 1.4, 3),
    (10000, 1, -2),
    (15000, 0.8, 5),
    (18000, 0.7, -4),
    (21000, 0.6, 2),
    (24000, 0.5, -1)
]

input_wav = "mug1.wav"
output_wav = "output.wav"

eq_audio = apply_parametric_eq(input_wav, eq_params)
eq_audio.export(output_wav, format="wav")
