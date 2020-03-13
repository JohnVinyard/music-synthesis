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

# Thursday, March 5, 2020 9PM
Least squares loss seems to work significantly better for FilterBank and 
Multiscale experiments

# Monday, March 9, 2020 7AM
Multi-headed FFT experiment produces very phase-y sounding audio.  Single-headed 
generator and discriminator pair worked much better.


# Monday, March 9, 2020 11AM
Flattening features from each band in the multiscale experiment is essential, 
otherwise lower bands are neglected by the generator.


# Wednesday, March 11, 2020 6AM
Despite some oddly promising early results, the DDSP-style generators seem to
be unstable and very difficult to train, never converging on satisfying-sounding
audio.  Putting these away (for now).  Shifting my attention toward conditional
discriminators next.

# Thursday, March 11, 2020 6AM
Using linear-spaced filters in both the generator and discriminator seems to 
help audio quality.  My hunch is that using mel-spaced oscillators for the 
generator greatly limits its ability to produce accurate frequencies in higher
ranges

# TODAY
- normal feature generator experiment using alternate audio repr
- autoregressive feature generator experiment
- autoregressive feature generator experiment using alternate audio repr


# TODO
- *try filterbank experiment with additional filtered noise channels*
- *try mel scale discriminator and linear-spaced generator*
- *try filterbank discriminator without multiscale*
- *try weight norm with FilterBank experiment*
- *try dilated convolutions in filterbank generator*


- try conditional discriminator

- try filterbank with log-scaled magnitudes, or a-weighting
- try filterbank with only small-scale


- think about what it means that multiscale formulation performs well on 
    this many speakers (i.e. it also performs well on TIMIT)
- try filterbank experiment with original MelGAN discriminator
- try weight norm with multiscale experiment


- organize spectrogram differently, with octaves grouped together
    - `(batch, channels, octave, f0, time)`
- try DDSP with FM synthesis or banks of octave filters

# Tools
- experiments configured in JSON file
- experiments that run some basic tests?

# Once I've Settled on the Best Model
- complete AWS deploy script
