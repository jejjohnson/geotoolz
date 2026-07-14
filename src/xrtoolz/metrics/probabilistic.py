"""Probabilistic / ensemble evaluation metrics — public re-export.

Layer-0 functions: :func:`spread_skill_ratio`, :func:`rank_histogram`,
:func:`ensemble_coverage`, :func:`reliability_curve`.

Layer-1 operators: :class:`SpreadSkillRatio`, :class:`RankHistogram`,
:class:`EnsembleCoverage`, :class:`ReliabilityCurve`.

Implementation lives in :mod:`xrtoolz.metrics._src.probabilistic`.
"""

from xrtoolz.metrics._src.probabilistic import (
    EnsembleCoverage,
    RankHistogram,
    ReliabilityCurve,
    SpreadSkillRatio,
    ensemble_coverage,
    rank_histogram,
    reliability_curve,
    spread_skill_ratio,
)


__all__ = [
    "EnsembleCoverage",
    "RankHistogram",
    "ReliabilityCurve",
    "SpreadSkillRatio",
    "ensemble_coverage",
    "rank_histogram",
    "reliability_curve",
    "spread_skill_ratio",
]
