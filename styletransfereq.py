import numpy as np
import torch
import torch.nn as nn
from subnetworks import mlp_architecture,encoder_architecture
from utils import load_wav_files
import random
import pandas as pd
from filters.peq import equalizer
from filters.proxy.proxyeq import load_saved_model as loadproxyeq
from filters.proxy.proxyeq import generate_random_gains
import soundfile as sf
import os
from filters.reverb import add_reverb
from filters.distortion import distortion


def custom_transform(x, min_val, max_val):
    s = torch.sigmoid(x/100000)
    return min_val + (s * (max_val - min_val))

class Styletransfer(nn.Module):
    def __init__(self, encoder_architecture, mlp_architecture, cnn_pretrained):
        super(Styletransfer, self).__init__()

        self.encoder1 = encoder_architecture()
        self.encoder2 = encoder_architecture()

        self.mlp = mlp_architecture()

        self.cnn = cnn_pretrained
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, input1a_stft, input2b_stft,input1a):
        encoded1 = self.encoder1(input1a_stft)
        encoded2 = self.encoder2(input2b_stft)

        concatenated = torch.cat((encoded1, encoded2), dim=1)
        concatenated = concatenated.view(concatenated.size(0), -1)
        input_dim = concatenated.size(-1) #229376
        mlp_output = self.mlp(concatenated) #estimated parameters
        mlp_output =custom_transform(mlp_output, -20, +20)
        concatenated2 = torch.cat((input1a, mlp_output), dim=1)
        cnn_output = self.cnn(concatenated2)

        return cnn_output,mlp_output
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


def randomfilter(audiopath):
    gains = generate_random_gains()
    processed_audio = equalizer(audiopath, gains)
    return processed_audio,gains
# Instantiate the custom model with appropriate encoder, MLP and CNN architectures
# encoder_architecture, mlp_architecture, cnn_pretrained should be replaced with your specific architectures

def train_network(input_directory,modelproxy_load_path,initial_model=None):
    # Prepare your inputs
    batch_size = 4
    input1_shape, input2_shape, output_shape = 1300, 1300, 1300
    learning_rate = 0.001
    epochs = 700
    losses_df = pd.DataFrame(columns=['epochs', 'MAE', 'MRSTFT'])
    fparam_df = pd.DataFrame(columns=['epochs', 'eq1','eq2','eq3','eq4','eq5','eq6',
                                      'eseq1','eseq2','eseq3','eseq4','eseq5','eseq6'])


    cnn_pretrained = loadproxyeq(modelproxy_load_path)
    if initial_model is None:
        model = Styletransfer(encoder_architecture, mlp_architecture, cnn_pretrained)
    else:
        model = initial_model
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    wav_files = load_wav_files(input_directory)


    for epoch in range(epochs):
        batch_size = 1
        for b in range(batch_size):
            input_wav_file = random.choice(wav_files)
            input1, sample_rate = sf.read(input_wav_file)
            filtered,filter_parameter = randomfilter(input_wav_file)
            input2 = filtered
            n_fft = 2048
            hop_length=256
            device = "cpu"
            input1_ts = torch.tensor(input1).float().to(device)
            input2_ts = torch.tensor(input2).float().to(device)
            input1a, input1b = torch.split(input1_ts, int(input1_ts.shape[0] / 2))
            input2a, input2b = torch.split(input2_ts, int(input2_ts.shape[0] / 2))
            input2a = input2a.unsqueeze(0)

            input1a_stft = (torch.stft(input1a, n_fft=n_fft, hop_length=hop_length,
                                 window=torch.hann_window(n_fft).to(device))).permute(2, 0, 1).unsqueeze(0)
            input2b_stft = (torch.stft(input2b, n_fft=n_fft, hop_length=hop_length,
                                 window=torch.hann_window(n_fft).to(device))).permute(2, 0, 1).unsqueeze(0)


        # Forward pass
        output,estimated_parameters = model(input1a_stft, input2b_stft,input1a.unsqueeze(0))

        # Calculate loss
        loss1 = mae_loss(output, input2a)
        loss2 = mr_stft_loss(output.squeeze(1).detach().numpy(), input2a.squeeze(1).detach().numpy(), sample_rate)
        loss = loss1 + loss2
        losses_df.loc[len(losses_df)] = [epoch, loss1.detach().numpy(), loss2]
        fparam_df.loc[len(fparam_df)] = [epoch]+filter_parameter+list(estimated_parameters.squeeze(0).detach().numpy())

        # Backpropagation
        loss.backward()

        # Update the weights of the MLP and Encoders using an optimizer
        optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
    return model,losses_df,fparam_df


def load_saved_model(model_load_path,modelproxy_load_path):
    model_state_dict = torch.load(model_load_path)
    loaded_model = Styletransfer(encoder_architecture, mlp_architecture, loadproxyeq(modelproxy_load_path))
    loaded_model.load_state_dict(model_state_dict)
    return loaded_model

def randomfilter_reverb(audiopath):
    # room scale : 0.1 - 1
    # wet level : 0.1 -0.8
    # dey level : 0.5 - 1
    gains = generate_random_gains()
    processed_audio = add_reverb(audiopath, '', room_scale=random.uniform(0.1,1), wet_level=random.uniform(0.1,0.8), dry_level=random.uniform(0.5,1))
    return processed_audio,gains

def randomfilter_distort(audiopath):
    gain = random.randint(1,20) #should be in 1-20
    threshold = random.uniform(0,1)#should be in 0-1
    gains = generate_random_gains()
    processed_audio = distortion(audiopath, '', gain, threshold)
    return processed_audio,gains

def Validate(input_directory,model):
    losses_df = pd.DataFrame(columns=['epochs', 'MAE', 'MRSTFT'])
    wav_files = load_wav_files(input_directory)
    for i in range(70):
        input_wav_file = random.choice(wav_files)
        input1, sample_rate = sf.read(input_wav_file)
        #filtered, filter_parameter = randomfilter_reverb(input_wav_file)#randomfilter(input_wav_file)
        filtered, filter_parameter = randomfilter_distort(input_wav_file)
        input2 = filtered
        n_fft = 2048
        hop_length = 256
        device = "cpu"
        input1_ts = torch.tensor(input1).float().to(device)
        input2_ts = torch.tensor(input2).float().to(device)
        input1a, input1b = torch.split(input1_ts, int(input1_ts.shape[0] / 2))
        input2a, input2b = torch.split(input2_ts, int(input2_ts.shape[0] / 2))
        input2a = input2a.unsqueeze(0)

        input1a_stft = (torch.stft(input1a, n_fft=n_fft, hop_length=hop_length,
                                   window=torch.hann_window(n_fft).to(device))).permute(2, 0, 1).unsqueeze(0)
        input2b_stft = (torch.stft(input2b, n_fft=n_fft, hop_length=hop_length,
                                   window=torch.hann_window(n_fft).to(device))).permute(2, 0, 1).unsqueeze(0)

        output, estimated_parameters = model(input1a_stft, input2b_stft, input1a.unsqueeze(0))

        # Calculate loss
        loss1 = mae_loss(output, input2a)
        loss2 = mr_stft_loss(output.squeeze(1).detach().numpy(), input2a.squeeze(1).detach().numpy(), sample_rate)
        losses_df.loc[len(losses_df)] = [i, loss1.detach().numpy(), loss2]
    return losses_df


if __name__ == "__main__":
    input_directory = "./data/raw5/"
    modelproxy_load_path ="./model/proxy/eq/30000"


    # Load the saved model
    if os.path.exists("./model/styletransfer/eq/1000"):
        model_load_path = "./model/styletransfer/eq/1300"
        print('load model')
        loaded_model = load_saved_model(model_load_path,modelproxy_load_path)
        validate = True
        if validate:
            losses_df_val = Validate(input_directory, loaded_model)
            losses_df_val.to_csv("./model/styletransfer/eq/losses_val_dist.csv", index=False)
            print("end of validation")
        # Continue training the model
        print("model loaded")
        model,losses,fparam_df = train_network(input_directory,modelproxy_load_path,initial_model=loaded_model)
    else:
        model,losses,fparam_df = train_network(input_directory,modelproxy_load_path,initial_model=None)

    # Save the continued model
    model_save_path = "./model/styletransfer/eq/2000"
    isExist = os.path.exists("./model/styletransfer/eq/")
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs("./model/styletransfer/eq/")
    torch.save(model.state_dict(), model_save_path)

    if os.path.exists("./model/styletransfer/eq/losses.csv"):
        df = pd.read_csv("./model/styletransfer/eq/losses.csv")
        newdf = pd.concat([df, losses])
        newdf.to_csv("./model/styletransfer/eq/losses.csv", index=False)

    else:
        losses.to_csv("./model/styletransfer/eq/losses.csv", index=False)

    if os.path.exists("./model/styletransfer/eq/fparam.csv"):
        df = pd.read_csv("./model/styletransfer/eq/fparam.csv")
        newdf = pd.concat([df, fparam_df])
        newdf.to_csv("./model/styletransfer/eq/fparam.csv", index=False)

    else:
        fparam_df.to_csv("./model/styletransfer/eq/fparam.csv", index=False)


