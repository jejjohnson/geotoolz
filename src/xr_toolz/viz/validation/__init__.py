"""V6 validation panels — terminal viz operators for V1–V5 outputs.

Per D15, these panels return :class:`matplotlib.figure.Figure`. They
are :class:`Operator` subclasses, so they slot into ``Sequential``
(as the last step) or ``Graph`` (as terminal nodes) alongside score
operators.

Shipped:

- :class:`LeadTimeSkillPanel`, :class:`ScaleSkillPanel`,
  :class:`SpectralSkillPanel` — V1 scale-skill outputs.
- :class:`EulerianLagrangianPanel` — V3 trajectories + Eulerian field.
- :class:`ProcessBudgetPanel` — V4 budget term breakdown.
- :class:`EventVerificationPanel` — V5 event match overlay +
  contingency stats.
"""

from xr_toolz.viz.validation._src.budgets import ProcessBudgetPanel
from xr_toolz.viz.validation._src.events import EventVerificationPanel
from xr_toolz.viz.validation._src.lagrangian import EulerianLagrangianPanel
from xr_toolz.viz.validation._src.palette import method_palette
from xr_toolz.viz.validation._src.psd import (
    PSDIsotropicPanel,
    PSDIsotropicScorePanel,
    PSDSpaceTimePanel,
    PSDSpaceTimeScorePanel,
)
from xr_toolz.viz.validation._src.regime_bars import RegionScoreBarPanel
from xr_toolz.viz.validation._src.rotary import RotaryPolarizationPanel
from xr_toolz.viz.validation._src.scales import (
    LeadTimeSkillPanel,
    ScaleSkillPanel,
    SpectralSkillPanel,
)
from xr_toolz.viz.validation._src.spatial import SpatialMapPanel


__all__ = [
    "EulerianLagrangianPanel",
    "EventVerificationPanel",
    "LeadTimeSkillPanel",
    "PSDIsotropicPanel",
    "PSDIsotropicScorePanel",
    "PSDSpaceTimePanel",
    "PSDSpaceTimeScorePanel",
    "ProcessBudgetPanel",
    "RegionScoreBarPanel",
    "RotaryPolarizationPanel",
    "ScaleSkillPanel",
    "SpatialMapPanel",
    "SpectralSkillPanel",
    "method_palette",
]
