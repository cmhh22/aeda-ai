"""Regulatory thresholds for sediment contamination (Buchman 2008 - NOAA SQuiRTs).

All concentrations are expressed in mg/kg (equivalent to ppm or ug/g) on a
dry weight basis.

Sources
-------
Buchman, M. F. (2008). NOAA Screening Quick Reference Tables, NOAA OR&R
Report 08-1. Marine Sediment section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MetalThresholds:
    """Regulatory thresholds for one metal in marine sediment (mg/kg)."""

    tel: Optional[float] = None
    pel: Optional[float] = None
    erl: Optional[float] = None
    erm: Optional[float] = None


TEL_PEL_MARINE_SEDIMENT: dict[str, MetalThresholds] = {
    "As": MetalThresholds(tel=7.24, pel=41.6, erl=8.2, erm=70.0),
    "Cd": MetalThresholds(tel=0.68, pel=4.21, erl=1.2, erm=9.6),
    "Cr": MetalThresholds(tel=52.3, pel=160.0, erl=81.0, erm=370.0),
    "Cu": MetalThresholds(tel=18.7, pel=108.0, erl=34.0, erm=270.0),
    "Hg": MetalThresholds(tel=0.13, pel=0.70, erl=0.15, erm=0.71),
    "Ni": MetalThresholds(tel=15.9, pel=42.8, erl=20.9, erm=51.6),
    "Pb": MetalThresholds(tel=30.24, pel=112.0, erl=46.7, erm=218.0),
    "Zn": MetalThresholds(tel=124.0, pel=271.0, erl=150.0, erm=410.0),
    "Ag": MetalThresholds(tel=0.73, pel=1.77, erl=1.0, erm=3.7),
    "Sb": MetalThresholds(tel=None, pel=None, erl=2.0, erm=25.0),
}


def get_thresholds(metal: str) -> MetalThresholds:
    """Retrieve marine-sediment thresholds for a given metal."""
    if metal not in TEL_PEL_MARINE_SEDIMENT:
        raise KeyError(
            f"Metal '{metal}' not found in NOAA marine-sediment thresholds. "
            f"Available: {sorted(TEL_PEL_MARINE_SEDIMENT.keys())}"
        )
    return TEL_PEL_MARINE_SEDIMENT[metal]
