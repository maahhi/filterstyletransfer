import torch
import torch.nn as nn

import torch.nn.functional as F


class EncoderArchitecture(nn.Module):
    def __init__(self, input_channels=2, conv_channels=32, output_channels=64):
        super(EncoderArchitecture, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(conv_channels, output_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        self.bn1 = nn.BatchNorm2d(conv_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class EncoderArchitectureold(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EncoderArchitecture, self).__init__()
        """input_dim = (1025,28,2)
        hidden_dim = (800,16,2)
        output_dim = (512,8,1)"""
        input_dim = 1025
        hidden_dim = 800
        output_dim = 512
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.n_fft = 2048
        self.hop_length = 256
        self.device = 'cpu'

    def forward(self, x):

        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                                  window=torch.hann_window(self.n_fft).to(self.device))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLPArchitecture(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPArchitecture, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Example usage:

# Encoder input dimensions
input1_dim = 128
input2_dim = 128

# Encoder hidden dimensions
encoder_hidden_dim = 256

# Encoder output dimensions
encoder_output_dim = 512

# MLP input dimensions (should be twice the size of the encoder output dimensions since we concatenate the two outputs)
mlp_input_dim = 229376

# MLP hidden dimensions
mlp_hidden_dim = 1024

# MLP output dimensions (should match the input dimensions of the pretrained CNN)
mlp_output_dim = 6  #6 for eq, 8 for eqcomp

encoder_architecture = lambda: EncoderArchitecture()
mlp_architecture = lambda: MLPArchitecture(mlp_input_dim, mlp_hidden_dim, mlp_output_dim)

