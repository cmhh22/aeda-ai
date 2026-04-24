# VERIFICACIÓN CIENTÍFICA — AEDA-AI Interpretation sobre ISOVIDA

## Objetivo

Ejecutar el script de diagnóstico extendido al final de este documento y **devolver el stdout completo** en la respuesta. No hace falta modificar código ni crear archivos — solo correr el script y pegar su salida íntegra en la respuesta.

El script imprime información en 6 secciones que permiten validar que:
1. Las unidades de las variables son correctas (metales traza en mg/kg, mayoritarios en %).
2. Los baselines seleccionados por el algoritmo son efectivamente las secciones más profundas de cada core.
3. Las clasificaciones TEL/PEL son científicamente plausibles para ISOVIDA (manglares de Cienfuegos — entorno moderadamente contaminado).
4. El Enrichment Factor produce valores razonables (no explota al infinito ni se queda todo en ~1.0).
5. No hay NaN excesivos en ningún paso.
6. Hay coherencia entre sitios (variación esperada, no artefactos numéricos).

## Instrucciones

1. Ejecutar el script con `python3 verify_interpretation.py` (o nombre equivalente) desde la raíz del proyecto.
2. Capturar el stdout completo.
3. Pegar la salida **tal cual** en la respuesta, dentro de un bloque de código, sin truncar nada.
4. Si algún paso arroja excepciones, pegar también el traceback completo.
5. No hace falta crear commit ni modificar archivos existentes.

---

## Script a ejecutar

```python
"""Extended verification of the interpretation module against ISOVIDA real data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeda.pipeline.runner import AEDAPipeline
from aeda.interpretation import (
    build_interpretation_report,
    compute_enrichment_factor,
    TEL_PEL_MARINE_SEDIMENT,
)


EXCLUDE = [
    "No", "Code", "Site_Name", "Pret_Code", "Código_muestra",
    "Sitio_muestreo", "Fecha_muestreo", "Core",
    "Latitud", "Longitud", "Profundidad",
]

DATA_PATH = "data/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx"

METALS = ["Pb", "Cr", "Cu", "Zn", "Ni", "As"]


def banner(text: str) -> None:
    print()
    print("=" * 72)
    print(f"  {text}")
    print("=" * 72)


# ---------- Load raw data ----------

banner("SECTION 1 — Raw data descriptive statistics")

pipeline = AEDAPipeline(impute_strategy="median")
results = pipeline.run(DATA_PATH, exclude_cols=EXCLUDE, sheet_name="DATA")
df = results.raw_data

print(f"Total samples: {len(df)}")
print(f"Unique sites:  {df['Sitio_muestreo'].nunique()}")
print(f"Sites:         {sorted(df['Sitio_muestreo'].dropna().unique().tolist())}")
print(f"Cores:         {sorted(df['Core'].dropna().unique().tolist())}")
print(f"Depth range:   {df['Profundidad'].min():.1f} – {df['Profundidad'].max():.1f} cm")

print("\nRaw concentration statistics for metals of interest + Al (reference):")
cols_to_describe = ["Al"] + METALS
available_cols = [c for c in cols_to_describe if c in df.columns]
stats_df = df[available_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
stats_df.columns = ["n", "mean", "std", "min", "median", "max"]
print(stats_df.round(3).to_string())

print("\nUnit sanity check:")
print("  If Al mean is around 5–10, units are likely %.")
print("  If Al mean is around 50000–100000, units are likely mg/kg.")
print("  For trace metals (Pb, Cr, Cu, etc.), mg/kg means values roughly in 1–500 range.")


# ---------- Baseline selection verification ----------

banner("SECTION 2 — Baseline selection per site (deepest sample)")

# Reproduce the baseline selection logic manually to verify
site_col, depth_col = "Sitio_muestreo", "Profundidad"
for site, sub in df.groupby(site_col):
    deepest_idx = sub[depth_col].idxmax()
    row = df.loc[deepest_idx]
    print(f"\n  Site: {site}")
    print(f"    Deepest sample: index={deepest_idx}, "
          f"depth={row[depth_col]:.1f} cm, core={row.get('Core','?')}")
    print(f"    Al = {row['Al']:.3f}")
    for m in METALS:
        if m in df.columns:
            print(f"    {m:3s} = {row[m]:.3f}")


# ---------- Run interpretation report ----------

banner("SECTION 3 — Interpretation report summary")

report = build_interpretation_report(
    df,
    metals=METALS,
    reference_element="Al",
    site_col="Sitio_muestreo",
    depth_col="Profundidad",
    baseline_strategy="deepest",
)
print(report.summary())


# ---------- TEL/PEL classification details ----------

banner("SECTION 4 — TEL/PEL classification details per metal")

for metal in METALS:
    if metal not in TEL_PEL_MARINE_SEDIMENT:
        print(f"\n  {metal}: (no NOAA thresholds)")
        continue
    thr = TEL_PEL_MARINE_SEDIMENT[metal]
    print(f"\n  {metal}: TEL={thr.tel}, PEL={thr.pel} mg/kg")
    if metal in report.tel_pel_classifications.columns:
        counts = report.tel_pel_classifications[metal].value_counts(dropna=False)
        total = counts.sum()
        for label, n in counts.items():
            pct = 100 * n / total if total else 0
            label_str = str(label) if pd.notna(label) else "NaN"
            print(f"    {label_str:20s}: {n:4d} ({pct:5.1f}%)")


# ---------- EF numeric distribution ----------

banner("SECTION 5 — Enrichment Factor distribution per metal")

if report.ef_result is not None:
    ef = report.ef_result.ef_values
    print(f"EF shape: {ef.shape}")
    print(f"EF NaN count per metal:")
    for m in METALS:
        if m in ef.columns:
            n_nan = int(ef[m].isna().sum())
            print(f"    {m}: {n_nan} NaN out of {len(ef)}")

    print("\nEF descriptive statistics (only non-NaN values):")
    ef_stats = ef[METALS].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    print(ef_stats[["count", "mean", "50%", "75%", "95%", "max"]].round(3).to_string())

    print("\nEF classification counts per metal:")
    for metal in METALS:
        if metal in report.ef_classifications.columns:
            counts = report.ef_classifications[metal].value_counts(dropna=False)
            total = counts.sum()
            print(f"\n  {metal}:")
            for label, n in counts.items():
                pct = 100 * n / total if total else 0
                label_str = str(label) if pd.notna(label) else "NaN"
                print(f"    {label_str:22s}: {n:4d} ({pct:5.1f}%)")
else:
    print("EF result is None — check for errors above.")


# ---------- Cross-tabulation: site × classification for Pb ----------

banner("SECTION 6 — Cross-tab: site × Pb classification")

pb_classes = report.tel_pel_classifications["Pb"]
pb_crosstab = pd.crosstab(
    df["Sitio_muestreo"], pb_classes, dropna=False
)
print("\nPb TEL/PEL classification by site:")
print(pb_crosstab.to_string())

if report.ef_result is not None:
    pb_ef_classes = report.ef_classifications["Pb"]
    pb_ef_crosstab = pd.crosstab(
        df["Sitio_muestreo"], pb_ef_classes, dropna=False
    )
    print("\nPb EF (Birch) classification by site:")
    print(pb_ef_crosstab.to_string())


# ---------- Quick surface-vs-deep comparison ----------

banner("SECTION 7 — Surface vs deep Pb for each site (anthropogenic signal check)")

for site, sub in df.groupby("Sitio_muestreo"):
    sub_sorted = sub.sort_values("Profundidad")
    if "Pb" not in sub_sorted.columns:
        continue
    top_3_shallow = sub_sorted.head(3)["Pb"].mean()
    bottom_3_deep = sub_sorted.tail(3)["Pb"].mean()
    ratio = top_3_shallow / bottom_3_deep if bottom_3_deep else float("nan")
    print(f"  {site:20s} | surface Pb mean (3 shallowest): {top_3_shallow:7.2f} mg/kg"
          f" | deep Pb mean (3 deepest): {bottom_3_deep:7.2f} mg/kg"
          f" | ratio: {ratio:.2f}")


banner("VERIFICATION SCRIPT COMPLETED")
```

---

## Qué debe contener la respuesta

La respuesta debe incluir **todo el stdout del script** dentro de un bloque de código. Ejemplo de estructura de la respuesta:

````
```
========================================================================
  SECTION 1 — Raw data descriptive statistics
========================================================================
Total samples: 273
...

========================================================================
  SECTION 2 — Baseline selection per site (deepest sample)
========================================================================
  Site: ...
...

(...y así hasta SECTION 7 y el banner final)
```
````

Si algún paso falla, pegar también el traceback completo de la excepción. No resumir, no editar, no truncar — la salida completa permite validar los resultados científicamente.
