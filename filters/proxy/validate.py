

input_wav_file = "../../data/raw/20.wav"  # Replace with the desired input WAV file path
audio_data, sample_rate = sf.read(input_wav_file)
gains = [-20,20,-20,-20,-20,-20]  # generate_random_gains()

input_data = process_audio(audio_data, gains)
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

output_tensor = continued_model(input_tensor)
output_audio_data = output_tensor.squeeze(1).detach().numpy()

#Save the output audio data to a new WAV file
output_file = "../../out20.wav"
sf.write(output_file, output_audio_data[0],samplerate=24000)