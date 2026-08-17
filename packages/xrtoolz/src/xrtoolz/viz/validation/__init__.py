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

Composable wrappers, which consume any of the above (or a plain
``(ds, ax) -> Any`` callable) and multiply it:

- :class:`FacetPanel` — one cell per value along a categorical dim.
- :class:`PairwiseComparePanel` — ``(ref, study, diff)`` triptych.
- :class:`AnimatePanel` + :func:`save_animation` — frames over a
  time-like dim. Not a panel: it yields a ``FuncAnimation``.
"""

from xrtoolz.viz.validation._src.animate import AnimatePanel, save_animation
from xrtoolz.viz.validation._src.budgets import ProcessBudgetPanel
from xrtoolz.viz.validation._src.compare import PairwiseComparePanel
from xrtoolz.viz.validation._src.events import EventVerificationPanel
from xrtoolz.viz.validation._src.facet import FacetPanel, seasonal_groupby
from xrtoolz.viz.validation._src.lagrangian import EulerianLagrangianPanel
from xrtoolz.viz.validation._src.palette import method_palette
from xrtoolz.viz.validation._src.psd import (
    PSDIsotropicPanel,
    PSDIsotropicScorePanel,
    PSDSpaceTimePanel,
    PSDSpaceTimeScorePanel,
)
from xrtoolz.viz.validation._src.regime_bars import RegionScoreBarPanel
from xrtoolz.viz.validation._src.rotary import RotaryPolarizationPanel
from xrtoolz.viz.validation._src.scales import (
    LeadTimeSkillPanel,
    ScaleSkillPanel,
    SpectralSkillPanel,
)
from xrtoolz.viz.validation._src.spatial import SpatialMapPanel


__all__ = [
    "AnimatePanel",
    "EulerianLagrangianPanel",
    "EventVerificationPanel",
    "FacetPanel",
    "LeadTimeSkillPanel",
    "PSDIsotropicPanel",
    "PSDIsotropicScorePanel",
    "PSDSpaceTimePanel",
    "PSDSpaceTimeScorePanel",
    "PairwiseComparePanel",
    "ProcessBudgetPanel",
    "RegionScoreBarPanel",
    "RotaryPolarizationPanel",
    "ScaleSkillPanel",
    "SpatialMapPanel",
    "SpectralSkillPanel",
    "method_palette",
    "save_animation",
    "seasonal_groupby",
]
