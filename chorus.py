import numpy as np
import soundfile as sf

def add_chorus(wav_file, output_file, depth=0.03, rate=1.5, mix=0.5):
    # Load the WAV file
    data, samplerate = sf.read(wav_file)

    # Initialize the chorus effect
    output_data = np.zeros_like(data)
    num_channels = data.shape[1] if data.ndim > 1 else 1
    delay_samples = int(depth * samplerate)
    phase = 0

    # Process the audio data
    for n in range(data.shape[0] - delay_samples):
        if num_channels == 1:
            output_data[n] = (1 - mix) * data[n] + mix * data[n + int(delay_samples * np.sin(phase))]
        else:
            for ch in range(num_channels):
                output_data[n][ch] = (1 - mix) * data[n][ch] + mix * data[n + int(delay_samples * np.sin(phase))][ch]

        # Update the phase
        phase += rate * 2 * np.pi / samplerate
        if phase > 2 * np.pi:
            phase -= 2 * np.pi

    # Save the processed data to a new WAV file
    sf.write(output_file, output_data, samplerate)

# Example usage:
add_chorus("mug1.wav", "output_chorus.wav", depth=0.03, rate=1.5, mix=0.5)
