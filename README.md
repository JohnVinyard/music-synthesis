# TODO
- try MelGAN with higher discriminator sampling rate - NOPE
- *try original formula with LJ Speech dataset*
- try original formula with different disc init
- try original formula with weight norm
- try MelGAN (as published) on LJ Speech
- try MelGAN (as published) on TIMIT (do shorter windows generalize better?)
- try original formula with TIMIT and conditional discriminator
- try filterbank discriminator original formula
- think about what it means that multiscale formulation performs well on 
    this many speakers
- try MelGAN with filter lengths that match given different sampling rate
- AWS training
- FFT models
- look at tensorboard
- organize spectrogram differently, with octaves grouped together
    - `(batch, channels, octave, f0, time)`

# Path to MelGAN
- different disc init
- weight norm in disc
- weight norm in generator
- higher frequency disc judgements
- higher frequency generator spec
- smaller training window (4096 samples)


# Tools
- conjure
- ~~ensure dimensions (better pad function)~~
- experiments configured in JSON file
- experiments that run some basic tests?


# Pain Points
- I'm having to recompute features
- I'm having to change network implementations
- I will have to change the code for previewing audio



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