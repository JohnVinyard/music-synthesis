# Saturday, Feb 22, 2020 11AM
Starting to investigate whether my choice of frequency scale is causing 
intelligibility problems with the speech I'm trying to generate.  Once I can 
produce clear and intelligible speech, it's time to revisit music and hip-hop 
datasets.

# Saturday, Feb 22, 2020 6PM
Introduce a report generator to assess how close generations are to original 
recordings

# Wednesday, Feb 26, 2020 7PM
Introduce a class to help me understand experiment parameters in an attempt to 
find the essential differences keeping me from landing on more intelligible 
speech.  Interestingly, the basic MelGAN generator fails to produce 
intelligible speech with my parameters, but the MultiScale model does much 
better.  It has significant phasing issues, but is much easier to understand.
The most obvious difference that could aid in speech intelligibility is that the
 original MelGAN experiment uses spectrograms sampled at ~80hz, while mine are
 only sampled at ~40hz.  I'll be testing the effects of this change first.

# Thursday, Feb 27, 2020 9AM
After ~12 hours of training, using spectrograms sampled at ~80hz still resulted
in less than intelligible speech.

# Monday, March 2, 2020 10AM
Trying out MelGAN code verbatim, and realized there was an important difference
between my initialization code and theirs:  I was also initializing biases with
normal distributions, which seemed to completely impede training.  Only 
initializing weights did the trick.  

# Wednesday, March 4, 2020 9AM
I'm now fairly confident I'm reproducing MelGAN results.  The last adjustment 
was to ensure that time-domain audio and spectorgrams were not being scaled
  independently.  The original MelGAN experiment also augments data by randomly
  varying each samples amplitude.  I'm skeptical about the contribution of this
  augmentation, but I've replicated it for now.  Next, moving on to trying out
  some of my own models with stronger audio priors.

# Thursday, March 5, 2020 9AM
I've now tried the multiscale and filterbank approaches with the LJSpeech 
dataset, with a 22050hz sample rate and short windows.  Neither performs as well
as the original MelGAN formulation, with the filter bank approach being the 
better of the two (speech is still muted).  It seems that the multiscale 
approach relies entirely on the low resolution FFT discriminator component, 
which explains the significant phase issues.  It's probably worth trying out
weight norm with both of these approaches 

# TODO
- try weight norm with FilterBank experiment
- try conditional discriminator
- try returning to least-squares GAN
- *try filterbank discriminator original formula on LJSpeech*
- tweak multiscale experiment's FFT parameters
- try weight norm with multiscale experiment
- FFT models
- try MelGAN (as published) on TIMIT (do shorter windows generalize better?)
- try original formula with TIMIT and conditional discriminator
- think about what it means that multiscale formulation performs well on 
    this many speakers
- look at tensorboard
- organize spectrogram differently, with octaves grouped together
    - `(batch, channels, octave, f0, time)`

# DONE
- ~~try multiscale experiment without low-res FFT component~~
    - This results in significantly less intelligible speech than the MelGAN
       formulation.  It should be removed from the winner's circle, for now.

# Tools
- experiments configured in JSON file
- experiments that run some basic tests?

# Once I've Settled on the Best Model
- complete AWS deploy script
