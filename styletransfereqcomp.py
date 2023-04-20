import numpy as np
import torch
import torch.nn as nn
from subnetworks import mlp_architecture,encoder_architecture
from utils import load_wav_files
import random
import pandas as pd
from filters.peq import equalizer
from filters.compressor import compressor
from filters.proxy.proxyeq import load_saved_model as loadproxyeq
from filters.proxy.proxycomp import load_saved_model as loadproxycomp
from filters.proxy.proxyeq import generate_random_gains
from filters.proxy.proxycomp import generate_random_parameter
import soundfile as sf
import os


class Styletransfer(nn.Module):
    def __init__(self, encoder_architecture, mlp_architecture, cnn1_pretrained, cnn2_pretrained):
        super(Styletransfer, self).__init__()

        self.encoder1 = encoder_architecture()
        self.encoder2 = encoder_architecture()

        self.mlp = mlp_architecture()

        self.cnn1 = cnn1_pretrained
        for param in self.cnn1.parameters():
            param.requires_grad = False
        self.cnn2 = cnn2_pretrained
        for param in self.cnn2.parameters():
            param.requires_grad = False

    def forward(self, input1a_stft, input2b_stft,input1a):
        encoded1 = self.encoder1(input1a_stft)
        encoded2 = self.encoder2(input2b_stft)

        concatenated = torch.cat((encoded1, encoded2), dim=1)
        concatenated = concatenated.view(concatenated.size(0), -1)
        input_dim = concatenated.size(-1) #229376
        mlp_output = self.mlp(concatenated) #estimated parameters : EQ gains + compressor parameters
        gains ,compparameters= torch.split(mlp_output,6,dim=1)
        concatenated2 = torch.cat((input1a, gains), dim=1)
        cnn1_output = self.cnn1(concatenated2)
        concatenated3 = torch.cat((cnn1_output, compparameters), dim=1)
        cnn2_output = self.cnn2(concatenated3)

        return cnn2_output,mlp_output
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
    comp_param = generate_random_parameter()
    EQed_audio = equalizer(audiopath, gains)

    processed_audio = compressor(EQed_audio,24000, comp_param[0], comp_param[1])
    return processed_audio,gains+comp_param

# Instantiate the custom model with appropriate encoder, MLP and CNN architectures
# encoder_architecture, mlp_architecture, cnn_pretrained should be replaced with your specific architectures
def train_network(input_directory,proxyeq_load_path,proxycomp_load_path,initial_model=None):
    # Prepare your inputs
    learning_rate = 0.001
    epochs = 450
    losses_df = pd.DataFrame(columns=['epochs', 'MAE', 'MRSTFT'])
    fparam_df = pd.DataFrame(columns=['epochs', 'eq1','eq2','eq3','eq4','eq5','eq6',
                                      'eseq1','eseq2','eseq3','eseq4','eseq5','eseq6',
                                      'cmpthreshold','cmpratio','escmpthreshold','escmpratio',])


    cnn1_pretrained = loadproxyeq(proxyeq_load_path)
    cnn2_pretrained = loadproxycomp(proxycomp_load_path)
    if initial_model is None:
        model = Styletransfer(encoder_architecture, mlp_architecture, cnn1_pretrained,cnn2_pretrained)
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
        if epoch % 20 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
    return model,losses_df,fparam_df


def load_saved_model(model_load_path,proxyeq_load_path,proxycomp_load_path):
    model_state_dict = torch.load(model_load_path)
    loaded_model = Styletransfer(encoder_architecture, mlp_architecture, loadproxyeq(proxyeq_load_path),loadproxycomp(proxycomp_load_path))
    loaded_model.load_state_dict(model_state_dict)
    return loaded_model

if __name__ == "__main__":
    input_directory = "../../data/raw"
    proxyeq_load_path = "./model/proxy/eq/30000"
    proxycomp_load_path = "./model/proxy/comp/20000"

    # Load the saved model
    if os.path.exists("./model/styletransfer/eqcmp/50"):
        model_load_path = "./model/styletransfer/eqcmp/50"

        loaded_model = load_saved_model(model_load_path, proxyeq_load_path,proxycomp_load_path)
        # Continue training the model
        print("model loaded")
        model,losses,fparam_df = train_network("./data/raw/", proxyeq_load_path,proxycomp_load_path, initial_model=loaded_model)
    else:
        model,losses,fparam_df = train_network("./data/raw/", proxyeq_load_path,proxycomp_load_path, initial_model=None)

    # Save the continued model
    model_save_path = "./model/styletransfer/eqcmp/500"
    isExist = os.path.exists("./model/styletransfer/eqcmp/")
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs("./model/styletransfer/eqcmp/")
    torch.save(model.state_dict(), model_save_path)

    if os.path.exists("./model/styletransfer/eqcmp/losses.csv"):
        df = pd.read_csv("./model/styletransfer/eqcmp/losses.csv")
        newdf = pd.concat([df, losses])
        newdf.to_csv("./model/styletransfer/eqcmp/losses.csv", index=False)

    else:
        losses.to_csv("./model/styletransfer/eqcmp/losses.csv", index=False)

    if os.path.exists("./model/styletransfer/eqcmp/fparam.csv"):
        df = pd.read_csv("./model/styletransfer/eqcmp/fparam.csv")
        newdf = pd.concat([df, fparam_df])
        newdf.to_csv("./model/styletransfer/eqcmp/fparam.csv", index=False)

    else:
        fparam_df.to_csv("./model/styletransfer/eqcmp/fparam.csv", index=False)


