"""Physical constants shared by the :mod:`xrtoolz.atm` primitives.

Values follow the WMO / CODATA 2018 conventions used by the plumax
reference implementations so the two stacks agree to round-off.
"""

from __future__ import annotations


#: Specific gas constant of dry air [J kg⁻¹ K⁻¹].
R_D: float = 287.04
#: Specific heat of dry air at constant pressure [J kg⁻¹ K⁻¹].
C_P: float = 1004.5
#: Poisson constant ``R_d / c_p`` (exponent of the Exner function).
KAPPA: float = R_D / C_P
#: Standard gravitational acceleration [m s⁻²].
G: float = 9.80665
#: Molar mass of dry air [kg mol⁻¹].
M_DRY: float = 0.0289644
#: Molar mass of methane [kg mol⁻¹].
M_CH4: float = 0.0160425
#: Avogadro constant [mol⁻¹].
N_A: float = 6.02214076e23
#: Boltzmann constant [J K⁻¹].
K_B: float = 1.380649e-23
#: Virtual-temperature coefficient: ``T_v = T (1 + EPSILON_TV q)``.
EPSILON_TV: float = 0.61
