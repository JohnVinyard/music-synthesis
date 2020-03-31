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

# Monday, March 23 2020 7AM
Conditional discriminators seem to be vital for music, or much more diverse 
sources of audio

# Tuesday, March 24 2020 10PM
Multiscale discriminators seem to perform fairly well on speech (see earlier 
results) and on music.  Finding where conditioning information should be 
injected in the discriminator is a work in progress, and whether transposed
convolutions are better than nearest-neighbor upsampling

# Wednesday, March 25 3PM
Injecting conditioning information both per-band and when all bands are fused
seems to help.

# Friday, March 27 2PM
An overfitting experiment seems to indicate that the multiscale representation
should not be decomposed and then recomposed during training.  Per-band 
gradient information seems to be much stronger in this case, which makes a lot of 
sense.

# Monday, March 30 8PM
After about 12 hours, the original MelGAN formulation captures the structure of 
audio, but produces fairly metallic sounds.
https://generation-report-realmelganexperiment.s3-us-west-1.amazonaws.com/index.html
Additionally, it appears I can get better high-frequency resolution using 128
PCA components versus a 128-band mel scale transformation.


 



# TODO
- how much better does filterbank gen and disc do, without any feature modifications?
    - try multiscale exeriment with filterbanks and filtered noise
- does more high-frequency info in features address high-band issue?
- PCA audio generation experiment
- do transposed convolutions or nearest-neighbor upsampling work better?
- does re/decompose approach address thin-sounding audio?
- does re/decompose cause missing middle issue?
- PCA-based feature generation experiment (possibly autoregressive)

- try per-channel embeddings in generator
- try per-channel filter sizes
- think about what it means that multiscale formulation performs well on 
    this many speakers (i.e. it also performs well on TIMIT)
- try DDSP with FM synthesis or banks of octave filters
