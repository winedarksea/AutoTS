# -*- coding: utf-8 -*-
"""
TVA — Time-varying Vectorized Architecture.

A modular forecasting graph that produces structurally consistent
forecasts across related time series via shared composite trend prototypes.
"""

from autots.evaluator.tva.tva import TVA
from autots.evaluator.tva.priors import SeriesMetadata, YggdrasilPriors
from autots.evaluator.tva.decomposition import NornDecomposer
from autots.evaluator.tva.structure import GraphSnapshot, StructureLearningConfig

__all__ = [
    'TVA',
    'SeriesMetadata',
    'YggdrasilPriors',
    'NornDecomposer',
    'GraphSnapshot',
    'StructureLearningConfig',
]
