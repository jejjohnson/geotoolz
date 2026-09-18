"""Product openers: one satellite L2 file -> one flat CF Dataset."""

from xrreader._src.local.openers.emit import open_emit_ch4_l2b
from xrreader._src.local.openers.ghgsat import open_ghgsat_ch4_l2
from xrreader._src.local.openers.tropomi import open_tropomi_ch4_l2


__all__ = ["open_emit_ch4_l2b", "open_ghgsat_ch4_l2", "open_tropomi_ch4_l2"]
