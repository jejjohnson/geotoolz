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
- :class:`CartesianMapPanel` — projection-free ``(y, x)`` map in km for
  idealised model output.
- :class:`EnergyTimeSeriesPanel` — model invariants (energy, enstrophy,
  …) over time.
- :class:`TimeSeriesErrorPanel` — per-timestep skill scores, one line per
  method.
- :class:`HovmollerPanel` — time × one spatial axis section of a field.

Composable wrappers, which consume any of the above (or a plain
``(ds, ax) -> Any`` callable) and multiply it:

- :class:`FacetPanel` — one cell per value along a categorical dim.
- :class:`PairwiseComparePanel` — ``(ref, study, diff)`` triptych.
- :class:`SpatialDiffPanel` — two-input ``(pred, ref, pred − ref)``
  snapshot triptych with shared field and zero-centred diff scales.
- :class:`AnimatePanel` + :func:`save_animation` — frames over a
  time-like dim. Not a panel: it yields a ``FuncAnimation``.
"""

from xrops.viz.validation._src.animate import AnimatePanel, save_animation
from xrops.viz.validation._src.budgets import ProcessBudgetPanel
from xrops.viz.validation._src.cartesian import CartesianMapPanel
from xrops.viz.validation._src.compare import PairwiseComparePanel
from xrops.viz.validation._src.diff import SpatialDiffPanel
from xrops.viz.validation._src.events import EventVerificationPanel
from xrops.viz.validation._src.facet import FacetPanel, seasonal_groupby
from xrops.viz.validation._src.hovmoller import HovmollerPanel
from xrops.viz.validation._src.lagrangian import EulerianLagrangianPanel
from xrops.viz.validation._src.palette import method_palette
from xrops.viz.validation._src.psd import (
    PSDIsotropicPanel,
    PSDIsotropicScorePanel,
    PSDSpaceTimePanel,
    PSDSpaceTimeScorePanel,
)
from xrops.viz.validation._src.regime_bars import RegionScoreBarPanel
from xrops.viz.validation._src.rotary import RotaryPolarizationPanel
from xrops.viz.validation._src.scales import (
    LeadTimeSkillPanel,
    ScaleSkillPanel,
    SpectralSkillPanel,
)
from xrops.viz.validation._src.spatial import SpatialMapPanel
from xrops.viz.validation._src.timeseries import (
    EnergyTimeSeriesPanel,
    TimeSeriesErrorPanel,
)


__all__ = [
    "AnimatePanel",
    "CartesianMapPanel",
    "EnergyTimeSeriesPanel",
    "EulerianLagrangianPanel",
    "EventVerificationPanel",
    "FacetPanel",
    "HovmollerPanel",
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
    "SpatialDiffPanel",
    "SpatialMapPanel",
    "SpectralSkillPanel",
    "TimeSeriesErrorPanel",
    "method_palette",
    "save_animation",
    "seasonal_groupby",
]
