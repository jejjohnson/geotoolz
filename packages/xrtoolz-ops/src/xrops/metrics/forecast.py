"""Forecast-skill (lead-time) evaluation metrics — public re-export.

Layer-0 functions: :func:`skill_by_lead_time`.

Layer-1 operators: :class:`SkillByLeadTime`.

Implementation lives in :mod:`xrops.metrics._src.forecast`.
"""

from xrops.metrics._src.forecast import SkillByLeadTime, skill_by_lead_time


__all__ = [
    "SkillByLeadTime",
    "skill_by_lead_time",
]
