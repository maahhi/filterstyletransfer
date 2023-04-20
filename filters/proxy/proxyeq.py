
import numpy as np
import soundfile as sf
import torch.optim as optim
from scipy.signal import stft
from filters.peq import equalizer
import pandas as pd

import torch
import torch.nn as nn

class EqualizerNN(nn.Module):
    def __init__(self, num_bands=6):
        super(EqualizerNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=9, stride=1, padding=4)
        self.conv4 = nn.Conv1d(16, 1, kernel_size=9, stride=1, padding=4)

        self.fc1 = nn.Linear(num_bands, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        audio_data_length = x.size(-1) - 6
        audio_data = x[:, :-6].unsqueeze(1)
        gains = x[:, -6:]

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


def generate_random_gains():
    return [random.randint(-20, 20) for _ in range(6)]


def process_audio(audio_data, gains):
    return np.concatenate((audio_data, np.array(gains)))


def load_saved_model(model_load_path):
    model_state_dict = torch.load(model_load_path)
    loaded_model = EqualizerNN()
    loaded_model.load_state_dict(model_state_dict)
    return loaded_model


def train_network(input_directory, equalizer_function, epochs=10000, learning_rate=0.001,initial_model=None):
    losses_df = pd.DataFrame(columns=['epochs','MAE','MRSTFT'])
    wav_files = load_wav_files(input_directory)

    if initial_model is None:
        model = EqualizerNN()
    else:
        model = initial_model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        input_wav_file = random.choice(wav_files)
        audio_data, sample_rate = sf.read(input_wav_file)
        gains = generate_random_gains()
        processed_audio = equalizer_function(input_wav_file, gains)

        input_data = process_audio(audio_data, gains)
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


def Validation(input_directory,model):
    losses_df = pd.DataFrame(columns=['epochs', 'MAE', 'MRSTFT'])
    wav_files = load_wav_files(input_directory)
    print('validation')

    for i in range(70):
        input_wav_file = random.choice(wav_files)
        audio_data, sample_rate = sf.read(input_wav_file)
        gains = generate_random_gains()
        processed_audio = equalizer(input_wav_file, gains)
        input_data = process_audio(audio_data, gains)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(processed_audio, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
        loss1 = mae_loss(target_tensor, output)
        loss2 = mr_stft_loss(target_tensor.squeeze(1).detach().numpy(), output.squeeze(1).detach().numpy(), sample_rate)
        losses_df.loc[len(losses_df)] = [i, loss1.detach().numpy(), loss2]
    input_wav_file = "../../data/raw5/202.wav"
    audio_data, sample_rate = sf.read(input_wav_file)
    output_file = "./202-r5.wav"
    sf.write(output_file, audio_data, samplerate=24000)
    Gains = [-10,-10,-10,-10,10,10]
    processed_audio = equalizer(input_wav_file, Gains)
    output_file = "./202-r5-filtered.wav"
    sf.write(output_file, processed_audio, samplerate=24000)
    input_data = process_audio(audio_data, Gains)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    output = model(input_tensor)
    output_file = "./202-r5-proxied.wav"
    sf.write(output_file, output.squeeze(1).detach().numpy()[0], samplerate=24000)
    return losses_df


if __name__ == "__main__":
    validation = True

    input_directory = "../../data/raw"

    # Load the saved model
    model_load_path = "../../model/proxy/eq/30000"
    loaded_model = load_saved_model(model_load_path)

    if validation:
        print("validation")
        losses_df_val = Validation("../../data/raw5",loaded_model)
        losses_df_val.to_csv("../../model/proxy/eq/losses_val.csv", index=False)
    else:
        # Continue training the model
        continued_model, lossesdf = train_network(input_directory, equalizer, initial_model=loaded_model)
        """
        continued_model, lossesdf = train_network(input_directory, equalizer, initial_model=None)
        """
        # Save the continued model
        model_save_path = "../../model/proxy/eq/40000"
        torch.save(continued_model.state_dict(), model_save_path)

        if os.path.exists("../../model/proxy/eq/losses.csv"):
            df = pd.read_csv("../../model/proxy/eq/losses.csv")
            newdf = pd.concat([df,lossesdf])
            newdf.to_csv("../../model/proxy/eq/losses.csv",index=False)

        else:
            lossesdf.to_csv("../../model/proxy/eq/losses.csv",index=False)

    """
        # Use the trained model
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
    """