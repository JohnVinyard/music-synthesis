from .fft import ComplexSTFTExperiment
from .mdct import GroupedMDCTExperiment
from .melgan import MultiScaleMelGanExperiment
from .realmelgan import RealMelGanExperiment
from .multiscale import \
    MultiScaleMultiResGroupedFeaturesExperiment, MultiScaleNoDeRecompose, \
    MultiScaleWithDDSPGenerator, MultiScaleNoDeRecomposeNoConvTranspose, \
    MultiScaleWithSTFTDiscriminator, MultiScaleNoDeRecomposeShortKernels, \
    MultiScaleNoDeRecomposeUnconditionedShortKernel, MultiScaleWithDeRecompose, \
    FilterBankMultiscaleExperiment
from .ddsp import OneDimDDSPExperiment
from .filterbank import \
    FilterBankExperiment, AlternateFilterBankExperiment, \
    ConditionalFilterBankExperiment
from .report import Report


from .featureexperiment import \
    TwoDimGeneratorFeatureExperiment, OneDimGeneratorFeatureExperiment, \
    OneDimGeneratorCollapseDiscriminatorFeatureExperiment, \
    NearestNeighborOneDimGeneratorCollapseDiscriminatorFeatureExperiment, \
    TwoDimFeatureExperiment, AutoregressiveFeatureExperiment, \
    PredictiveFeatureExperiment
