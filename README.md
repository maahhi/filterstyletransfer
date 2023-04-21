# filter style transfer

Synthesizer final project: style transfer

Mahya Khazaei

programming language : python

framework : lightning pytorch

This project gonna be based on [Style Transfer of Audio Effects with Differentiable Signal Processing](https://arxiv.org/abs/2207.08759). As a short summary about it, I can say this paper has done a unified implementation of a different way to pass gradient from a DSP system (differentiable audio effects).
![image](https://user-images.githubusercontent.com/23381605/230758182-b30fc5fe-020d-4a42-bcd5-568de0db75a6.png)


The options for DAE are: by network proxies, by pytorch implementation of the filter, and by numerical gradient approximation.
Despite their considerable result, they only used a parametric EQ and a compressor.

![image](https://user-images.githubusercontent.com/23381605/230758312-a7dbb5e2-2dcc-4abb-a87f-58e2b08bba53.png)

You can see results on [this page](https://csteinmetz1.github.io/DeepAFx-ST/) and [explaining video from youtube](https://www.youtube.com/watch?v=-ezTdjRpAvw&t=2204s&ab_channel=Music%2BAIReadingGroup)(this video is different from the video on the previous page)

In this study, the project was implemented from scratch, beginning with the implementation of various parametric DSPs as filters, including Parametric Equalizer (PEQ), Compressor (CMP), Reverb, and Distortion. The PEQ was designed with six bands, ranging from 20 to 1200 Hz, and allowed the user to pass six different gains from -20 to 20 as the filter parameters. The compressor was simplified to accept only threshold (-20, 20) and ratio (2, 6) as input parameters, while the reverb accepted room scale from 0.1 to 1, wet level from 0.1 to 0.8, and dry level from 0.5 to 1.

you can find related files in `filters` directory.

To employ full neural proxy hybrids, a proxy was trained for each filter used within the architecture to mimic the filter and enable optimizer passage. CNN networks were utilized to train both the PEQ and CMP proxies. These proxies were integrated as a differentiable audio effect component within a pipeline. The component accepted input audio and eight parameters (six for the PEQ and two for the CMP), applying the PEQ proxy with the first six parameters on the input audio to generate the PEQ output. Subsequently, the CMP proxy applied compression based on the last two parameters to the PEQ output, producing the final output.

you can find related files in `filters/proxy` directory.

The model architecture employed CNN as encoders and MLP for the controller. The majority of neural network layers in this project utilized ReLU activation functions, while the controller's final output, which determined the parameter, employed a sigmoid and scaler to facilitate training by providing parameter range hints.

related files are `styletransfereq.py` and `styletransfereqcomp.py`

A subset of the train-clean-360 dataset, which is part of LibriTTS, was used for this project. It was divided into 1,000 audio tracks with a length of 0.53 seconds and a sample rate of 24,000. The time domain loss was computed using the mean absolute error (MAE), while the frequency domain loss was calculated using the multi-resolution short-time Fourier transform loss (MR-STFT). The MR-STFT loss is the sum of the distances between the STFT of the ground truth and estimated waveforms, measured in both log and linear domains across multiple resolutions, with window sizes W ∈ [32, 128, 512, 2048] and hop sizes H = 256.

you can find loss result and trained model of proxies in `model/proxy` and loss for style transfer in `model/styletransfer`