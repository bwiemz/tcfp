"""TCFP training utilities submodule.

Provides utilities for managing the training lifecycle of TCFP models:
  - Checkpointing (TCFPCheckpointer, ShieldWorldArchiver)
  - Health monitoring (TCFPMonitor, GradientCorruptionDetector)
  - Curriculum and phase detection (CUSUMPhaseDetector, ProgressiveQuantizer,
    QuantizationCurriculum)
  - Layer policies (apply_highway_routing, AdaptiveABDController,
    FisherSensitivityMap, GradientBiasCorrector, MomentumAlignmentTracker)
  - Config presets (TrainingPreset)
"""

from __future__ import annotations

from tcfp.training.checkpointing import ShieldWorldArchiver, TCFPCheckpointer
from tcfp.training.curriculum import (
    CUSUMPhaseDetector,
    PhaseConfig,
    ProgressiveQuantizer,
    QuantizationCurriculum,
    WaveConfig,
)
from tcfp.training.monitoring import (
    GradientCorruptionDetector,
    GradientStatus,
    TCFPMonitor,
    TrainingAlert,
)
from tcfp.training.policy import (
    AdaptiveABDController,
    FisherSensitivityMap,
    GradientBiasCorrector,
    HysteresisState,
    MomentumAlignmentState,
    MomentumAlignmentTracker,
    apply_highway_routing,
)
from tcfp.training.presets import TrainingPreset

__all__ = [
    "TCFPCheckpointer",
    "ShieldWorldArchiver",
    "TCFPMonitor",
    "TrainingAlert",
    "GradientCorruptionDetector",
    "GradientStatus",
    "CUSUMPhaseDetector",
    "ProgressiveQuantizer",
    "PhaseConfig",
    "QuantizationCurriculum",
    "WaveConfig",
    "apply_highway_routing",
    "AdaptiveABDController",
    "HysteresisState",
    "FisherSensitivityMap",
    "GradientBiasCorrector",
    "MomentumAlignmentState",
    "MomentumAlignmentTracker",
    "TrainingPreset",
]
