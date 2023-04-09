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

What have I done till now?
Reading papers and understanding the architecture
Read source code and shallow understanding of it.

What I planned to do?
As they only applied EQ and compressor, it is interesting for me to see how good is the proposed model for other filters like distortion and reverb. For this research, I will apply other filter1 to audio1 and audio2 then I’ll ask different models to transfer the style of filtered(audio1) to origins audio2 and will do different measurements between the system result and filtered(audio2) so I can see if the EQ and compressor are better in doing other filters or not.


For bones, I can implement a minimal version of the whole system with just one of DAE from scratch by myself and also limited DSP application for the training phase and show improvement in accuracy.

I’m eager for full marks for my project, please let me know if these are enough. Please let me know if you have any ideas about what I can do for this project.


