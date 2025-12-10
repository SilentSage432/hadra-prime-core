# prime_core package
# Influence Substrate Kernel for ADRAE Prime-Core

from .influence_substrate import (
    InfluenceSubstrateKernel,
    A130_SubstrateCouplingGate,
    A131_DriftAttenuationLayer,
    A132_CurvatureHarmonizationLayer,
    A133_MultiScaleAlignmentLayer,
    A134_SpectralEqualizationLayer,
    A135_EnergyNormalizationLayer,
    A136_CrossManifoldAlignmentRegulator,
    MFPhaseBase,
    normalize_tensor,
)

__all__ = [
    'InfluenceSubstrateKernel',
    'A130_SubstrateCouplingGate',
    'A131_DriftAttenuationLayer',
    'A132_CurvatureHarmonizationLayer',
    'A133_MultiScaleAlignmentLayer',
    'A134_SpectralEqualizationLayer',
    'A135_EnergyNormalizationLayer',
    'A136_CrossManifoldAlignmentRegulator',
    'MFPhaseBase',
    'normalize_tensor',
]

