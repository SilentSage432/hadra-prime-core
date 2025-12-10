# prime_core package
# Influence Substrate Kernel for ADRAE Prime-Core

from .influence_substrate import (
    InfluenceSubstrateKernel,
    A130_SubstrateCouplingGate,
    A131_DriftAttenuationLayer,
    A132_CurvatureHarmonizationLayer,
    A133_MultiScaleAlignmentLayer,
    MFPhaseBase,
    normalize_tensor,
)

__all__ = [
    'InfluenceSubstrateKernel',
    'A130_SubstrateCouplingGate',
    'A131_DriftAttenuationLayer',
    'A132_CurvatureHarmonizationLayer',
    'A133_MultiScaleAlignmentLayer',
    'MFPhaseBase',
    'normalize_tensor',
]

