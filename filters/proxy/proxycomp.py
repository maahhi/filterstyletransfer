
import numpy as np
import soundfile as sf
import torch.optim as optim
from scipy.signal import stft
from filters.compressor import compressor
import pandas as pd

import torch
import torch.nn as nn

class CompressorNN(nn.Module):
    def __init__(self, num_bands=2):
        super(CompressorNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=9, stride=1, padding=4)
        self.conv4 = nn.Conv1d(16, 1, kernel_size=9, stride=1, padding=4)

        self.fc1 = nn.Linear(num_bands, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        p = 2
        audio_data_length = x.size(-1) - p
        audio_data = x[:, :-p].unsqueeze(1)
        gains = x[:, -p:]

        x = torch.relu(self.conv1(audio_data))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)

        gains = torch.relu(self.fc1(gains))
        gains = torch.sigmoid(self.fc2(gains))
        gains_expanded = gains.unsqueeze(-1).expand_as(x)

        x = x * gains_expanded
        return x.squeeze(1)

def mae_loss(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


def mr_stft_loss(audio1, audio2,sample_rate, n_fft_list=[32,128,512,2048], hop_length=256, device="cpu"):

    loss = 0
    audio1_tensor = torch.tensor(audio1).float().to(device)
    audio2_tensor = torch.tensor(audio2).float().to(device)

    for n_fft in n_fft_list:
        stft_audio1 = torch.stft(audio1_tensor, n_fft=n_fft, hop_length=hop_length,
                                 window=torch.hann_window(n_fft).to(device))
        stft_audio2 = torch.stft(audio2_tensor, n_fft=n_fft, hop_length=hop_length,
                                 window=torch.hann_window(n_fft).to(device))

        magnitude_audio1 = torch.sqrt(torch.sum(stft_audio1 ** 2, dim=-1))
        magnitude_audio2 = torch.sqrt(torch.sum(stft_audio2 ** 2, dim=-1))

        loss += torch.mean(torch.abs(magnitude_audio1 - magnitude_audio2)).item()

    return loss / len(n_fft_list)


import os
import random


def load_wav_files(path):
    wav_files = []
    for file in os.listdir(path):
        if file.endswith(".wav"):
            wav_files.append(os.path.join(path, file))
    return wav_files


def generate_random_parameter():
    return [random.randint(-20, 20),random.randint(2, 6)]


def process_audio(audio_data, parameter):
    return np.concatenate((audio_data, np.array(parameter)))


def load_saved_model(model_load_path):
    model_state_dict = torch.load(model_load_path)
    loaded_model = CompressorNN()
    loaded_model.load_state_dict(model_state_dict)
    return loaded_model


def train_network(input_directory, compressor_function, epochs=10000, learning_rate=0.001,initial_model=None):
    losses_df = pd.DataFrame(columns=['epochs','MAE','MRSTFT'])
    wav_files = load_wav_files(input_directory)

    if initial_model is None:
        model = CompressorNN()
    else:
        model = initial_model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        input_wav_file = random.choice(wav_files)
        audio_data, sample_rate = sf.read(input_wav_file)
        parameters = generate_random_parameter()
        processed_audio = compressor_function(audio_data,sample_rate, threshold=parameters[0],ratio=parameters[1])

        input_data = process_audio(audio_data, parameters)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(processed_audio, dtype=torch.float32).unsqueeze(0)

        optimizer.zero_grad()
        output = model(input_tensor)
        loss1 = mae_loss(target_tensor, output)
        loss2 = mr_stft_loss(target_tensor.squeeze(1).detach().numpy(), output.squeeze(1).detach().numpy(), sample_rate)
        losses_df.loc[len(losses_df)]= [epoch,loss1.detach().numpy(),loss2]
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
    return model,losses_df

if __name__ == "__main__":

    input_directory = "../../data/raw"

    # Load the saved model
    model_load_path = "../../model/proxy/comp/10000"
    loaded_model = load_saved_model(model_load_path)

    # Continue training the model
    continued_model,lossesdf = train_network(input_directory, compressor, initial_model=loaded_model)
    """
    model, lossesdf = train_network(input_directory, compressor, initial_model=None)
    """
    # Save the continued model
    model_save_path = "../../model/proxy/comp/20000"
    torch.save(continued_model.state_dict(), model_save_path)

    if os.path.exists("../../model/proxy/comp/losses.csv"):
        df = pd.read_csv("../../model/proxy/comp/losses.csv")
        newdf = pd.concat([df,lossesdf])
        newdf.to_csv("../../model/proxy/comp/losses.csv",index=False)

    else:
        lossesdf.to_csv("../../model/proxy/comp/losses.csv",index=False)


    # Use the trained model
    input_wav_file = "../../data/raw/20.wav"  # Replace with the desired input WAV file path
    audio_data, sample_rate = sf.read(input_wav_file)
    params = [-20,4]  # generate_random_gains()

    input_data = process_audio(audio_data, params)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    output_tensor = continued_model(input_tensor)
    output_audio_data = output_tensor.squeeze(1).detach().numpy()

    #Save the output audio data to a new WAV file
    output_file = "../../out20.wav"
    sf.write(output_file, output_audio_data[0],samplerate=24000)