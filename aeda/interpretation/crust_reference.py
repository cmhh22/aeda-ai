"""Upper continental crust reference values from Rudnick and Gao (2013).

These values represent typical concentrations of major and trace elements
in the upper continental crust. They are used as a complementary reference
to assess whether observed concentrations are within typical natural
background ranges. They are NOT a quality threshold like TEL or PEL;
they only describe what is geochemically common in continental crust.

Reference
---------
Rudnick, R.L. and Gao, S. (2013) "Composition of the Continental Crust",
in Holland, H.D. and Turekian, K.K. (eds.) Treatise on Geochemistry,
2nd ed., vol. 4. Oxford: Elsevier, pp. 1-51.
https://doi.org/10.1016/B978-0-08-095975-7.00301-6

All values in mg/kg (ppm) for trace elements and weight percent (wt%) for majors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrustReferenceValue:
    """A single element's reference value in the upper continental crust."""

    value: float
    unit: str  # either "wt%" or "mg/kg"
    notes: str = ""


# Values from Rudnick & Gao (2013), Table 3 — Upper Continental Crust composition.
UPPER_CONTINENTAL_CRUST: dict[str, CrustReferenceValue] = {
    # Major elements (wt%)
    "Si": CrustReferenceValue(value=31.10, unit="wt%", notes="SiO2 66.62 wt%"),
    "Al": CrustReferenceValue(value=8.15, unit="wt%", notes="Al2O3 15.40 wt%"),
    "Fe": CrustReferenceValue(value=3.92, unit="wt%", notes="Fe2O3T 5.04 wt%"),
    "Ca": CrustReferenceValue(value=2.57, unit="wt%", notes="CaO 3.59 wt%"),
    "Na": CrustReferenceValue(value=2.43, unit="wt%", notes="Na2O 3.27 wt%"),
    "K": CrustReferenceValue(value=2.32, unit="wt%", notes="K2O 2.80 wt%"),
    "Mg": CrustReferenceValue(value=1.50, unit="wt%", notes="MgO 2.48 wt%"),
    "Ti": CrustReferenceValue(value=0.38, unit="wt%", notes="TiO2 0.64 wt%"),
    "P": CrustReferenceValue(value=0.066, unit="wt%", notes="P2O5 0.15 wt%"),
    "Mn": CrustReferenceValue(value=0.0775, unit="wt%", notes="MnO 0.10 wt%"),
    # Trace elements (mg/kg)
    "As": CrustReferenceValue(value=4.8, unit="mg/kg"),
    "Ba": CrustReferenceValue(value=628.0, unit="mg/kg"),
    "Cd": CrustReferenceValue(value=0.09, unit="mg/kg"),
    "Co": CrustReferenceValue(value=17.3, unit="mg/kg"),
    "Cr": CrustReferenceValue(value=92.0, unit="mg/kg"),
    "Cs": CrustReferenceValue(value=4.9, unit="mg/kg"),
    "Cu": CrustReferenceValue(value=28.0, unit="mg/kg"),
    "Ga": CrustReferenceValue(value=17.5, unit="mg/kg"),
    "Hg": CrustReferenceValue(value=0.05, unit="mg/kg"),
    "Li": CrustReferenceValue(value=24.0, unit="mg/kg"),
    "Mo": CrustReferenceValue(value=1.1, unit="mg/kg"),
    "Nb": CrustReferenceValue(value=12.0, unit="mg/kg"),
    "Ni": CrustReferenceValue(value=47.0, unit="mg/kg"),
    "Pb": CrustReferenceValue(value=17.0, unit="mg/kg"),
    "Rb": CrustReferenceValue(value=84.0, unit="mg/kg"),
    "Sb": CrustReferenceValue(value=0.4, unit="mg/kg"),
    "Sc": CrustReferenceValue(value=14.0, unit="mg/kg"),
    "Sn": CrustReferenceValue(value=2.1, unit="mg/kg"),
    "Sr": CrustReferenceValue(value=320.0, unit="mg/kg"),
    "Th": CrustReferenceValue(value=10.5, unit="mg/kg"),
    "U": CrustReferenceValue(value=2.7, unit="mg/kg"),
    "V": CrustReferenceValue(value=97.0, unit="mg/kg"),
    "Y": CrustReferenceValue(value=21.0, unit="mg/kg"),
    "Zn": CrustReferenceValue(value=67.0, unit="mg/kg"),
    "Zr": CrustReferenceValue(value=193.0, unit="mg/kg"),
}


def get_crust_reference(element: str) -> CrustReferenceValue:
    """Retrieve the upper continental crust reference value for an element."""
    if element not in UPPER_CONTINENTAL_CRUST:
        raise KeyError(
            f"Element '{element}' not found in Rudnick & Gao (2013) table. "
            f"Available: {sorted(UPPER_CONTINENTAL_CRUST.keys())}"
        )
    return UPPER_CONTINENTAL_CRUST[element]


def compare_to_crust(concentration: float, element: str, sample_unit: str = "mg/kg") -> dict:
    """Compare a measured concentration to the upper continental crust value."""
    ref = get_crust_reference(element)
    crust_value = ref.value
    if ref.unit == "wt%" and sample_unit == "mg/kg":
        crust_value = ref.value * 10000
    elif ref.unit == "mg/kg" and sample_unit == "wt%":
        crust_value = ref.value / 10000

    if crust_value == 0:
        return {"ratio": float("nan"), "label": "undefined"}

    ratio = concentration / crust_value
    if ratio < 0.5:
        label = "below_crust"
    elif ratio <= 2:
        label = "similar_to_crust"
    elif ratio <= 10:
        label = "enriched"
    else:
        label = "highly_enriched"

    return {
        "ratio": ratio,
        "label": label,
        "crust_value": crust_value,
        "crust_unit": ref.unit,
    }