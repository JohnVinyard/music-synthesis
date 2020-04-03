In this repo, I'm developing models for a two-stage music synthsis pipeline.

1. A generative model that can produce sequences of low-frequency audio features, 
   such as a mel spectrogram, or a sequence of chroma and MFCC features.
2. A conditional generative model that can produce raw audio from the 
   low-frequency features.

The second stage is inspired by papers developing spectrogram-to-speech vocoders 
such as:
 
- [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711)
- [High Fidelity Speech Synthesis with Adversarial Networks](https://arxiv.org/abs/1909.11646)

You can read more about early experiments developing models for the second 
stage in [this blog post](https://johnvinyard.github.io/synthesis/2020/04/01/gan-vocoder.html). 