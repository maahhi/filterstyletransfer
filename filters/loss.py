import librosa
import torch
import numpy as np
import pandas as pd

def load_audio(audio_file, sr=24000):
    audio, _ = librosa.load(audio_file, sr=sr)
    return audio

def calculate_mae(audio1, audio2):
    return np.mean(np.abs(audio1 - audio2))

def compute_mr_stft_loss(audio1, audio2, n_fft_list=[512, 1024, 2048], hop_length=256, device="cpu"):
    loss = 0
    audio1_tensor = torch.tensor(audio1).float().to(device)
    audio2_tensor = torch.tensor(audio2).float().to(device)

    for n_fft in n_fft_list:
        stft_audio1 = torch.stft(audio1_tensor, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).to(device))
        stft_audio2 = torch.stft(audio2_tensor, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft).to(device))

        magnitude_audio1 = torch.sqrt(torch.sum(stft_audio1 ** 2, dim=-1))
        magnitude_audio2 = torch.sqrt(torch.sum(stft_audio2 ** 2, dim=-1))

        loss += torch.mean(torch.abs(magnitude_audio1 - magnitude_audio2)).item()

    return loss / len(n_fft_list)

def main(file1,file2):
    audio_file1 = file1
    audio_file2 = file2

    audio1 = load_audio(audio_file1)
    audio2 = load_audio(audio_file2)

    mae = calculate_mae(audio1, audio2)
    mr_stft_loss = compute_mr_stft_loss(audio1, audio2)

    print(f"Mean Absolute Error: {mae}")
    print(f"Multi-Resolution STFT Loss: {mr_stft_loss}")
    return mae,mr_stft_loss

if __name__ == "__main__":
    f1 = ['1.wav','1peq.wav','1quiet.wav']#['mug1.wav','output_reverb.wav','output_distorted.wav','output_chorus.wav']
    f2 = ['1.wav','1peq.wav','1quiet.wav']#['mug1.wav','output_reverb.wav','output_distorted.wav','output_chorus.wav']
    for i in f1:
        for j in f2:
            print(i,j)
            main(i,j)
