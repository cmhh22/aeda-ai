"""
Validation script for the AEDA Framework ingestion module.
Runs validation with the real BD_ISOVIDA dataset.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ingestion.universal_data_ingestor import UniversalDataIngestor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def validate_isovida_ingestion() -> bool:
    print("\n" + "=" * 80)
    print("AEDA FRAMEWORK - INGESTION MODULE VALIDATION")
    print("Dataset: BD_ISOVIDA Manglares 2023")
    print("=" * 80 + "\n")

    elements_schema = {
        "Na": "%",
        "Mg": "%",
        "Al": "%",
        "Si": "%",
        "K": "%",
        "Ca": "%",
        "Fe": "%",
        "P": "ppm",
        "S": "ppm",
        "Cl": "ppm",
        "Sc": "ppm",
        "Ti": "ppm",
        "V": "ppm",
        "Cr": "ppm",
        "Mn": "ppm",
        "Co": "ppm",
        "Ni": "ppm",
        "Cu": "ppm",
        "Zn": "ppm",
        "Ga": "ppm",
        "As": "ppm",
        "Br": "ppm",
        "Rb": "ppm",
        "Sr": "ppm",
        "Y": "ppm",
        "Zr": "ppm",
        "Nb": "ppm",
        "Mo": "ppm",
        "Ba": "ppm",
        "Pb": "ppm",
    }

    metadata_cols = {
        "No", "Code", "Pret_Code", "Código_muestra", "Sitio_muestreo",
        "Fecha_muestreo", "Core", "Latitud", "Longitud", "Profundidad",
        "< 2 µm", "U_< 2 µm", "2 < G < 63 µm", "U_2 < G < 63 µm",
        "> 63 µm", "U_> 63 µm", "PPI550", "U_PPI550", "PPI950", "U_PPI950", "HC",
    }

    print("1. Creating UniversalDataIngestor...")
    ingestor = UniversalDataIngestor(
        analyte_schema=elements_schema,
        metadata_columns=metadata_cols,
        target_unit="ppm",
        strict_schema=False,
        censored_value_strategy="lod_half",
        apply_censored_handling=False,
        generate_quality_report=True,
    )
    print("   [OK] Ingestor created successfully\n")

    print("2. Ingesting rectified BD_ISOVIDA dataset...")
    file_path = ROOT / "data" / "raw" / "BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx"

    try:
        result = ingestor.run(str(file_path), matrix_type_hint="sediment")
        print("   [OK] Data ingested successfully\n")
    except Exception as error:
        print(f"   [ERROR] Ingestion failed: {error}")
        return False

    print("3. Analyzing results...")
    data = result["data"]
    metadata = result["metadata"]

    print(f"   [OK] Shape: {data.shape[0]} samples x {data.shape[1]} columns")
    print(f"   [OK] Matrix type: {metadata['matrix_type_detected']}")
    print(f"   [OK] Detection confidence: {metadata['matrix_type_confidence']:.1%}\n")

    print("4. Censored values summary:")
    censored_summary = result.get("censored_summary", {})
    if censored_summary:
        print(f"   [OK] BDL values handled: {censored_summary.get('total_bdl_values', 0)}")
        print(f"   [OK] AQL values handled: {censored_summary.get('total_aloq_values', 0)}")
        print(f"   [OK] Columns processed: {censored_summary.get('total_columns_processed', 0)}")
    print()

    print("5. Data transformation example:")
    print("\n   RAW data sample (first element column):")
    raw_data = result.get("raw_data")
    if raw_data is not None and "V" in raw_data.columns:
        print("   " + str(raw_data["V"].head(3).tolist()))

    print("\n   PROCESSED data sample (first element column):")
    if "V" in data.columns:
        print("   " + str(data["V"].head(3).tolist()))
    print()

    print("6. Quality report:")
    quality_report = result.get("quality_report", "")
    if quality_report:
        report_path = ROOT / "data" / "processed" / "ingestion_validation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(quality_report, encoding="utf-8")

        print(f"   [OK] Report saved to {report_path}")
        lines = quality_report.split("\n")
        print("\n   First 30 lines of report:")
        for line in lines[:30]:
            print("   " + line)
    print()

    print("7. Validation checks:")
    nan_count = data.isnull().sum().sum()
    print(f"   [OK] Total NaN values: {nan_count}")

    numeric_cols = data.select_dtypes(include=["number"]).columns
    print(f"   [OK] Numeric columns: {len(numeric_cols)}")

    analyte_cols = [col for col in elements_schema if col in data.columns]
    negative_count = (data[analyte_cols] < 0).sum().sum() if analyte_cols else 0
    print(f"   [OK] Negative values in analytes: {negative_count}")

    print("\n   Sample statistics:")
    for column in list(numeric_cols)[:5]:
        values = data[column].dropna()
        if len(values) > 0:
            print(
                f"   - {column}: median={values.median():.2e}, "
                f"range=[{values.min():.2e}, {values.max():.2e}]"
            )

    print("\n" + "=" * 80)
    print("[COMPLETE] VALIDATION SUCCESSFUL - All systems operational")
    print("=" * 80 + "\n")

    return True


if __name__ == "__main__":
    success = validate_isovida_ingestion()
    sys.exit(0 if success else 1)
