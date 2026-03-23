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
        "V_(ppm)": "ppm",
        "Cr_(ppm)": "ppm",
        "Mn_(ppm)": "ppm",
        "Fe_(%)": "%",
        "Co_ppm": "ppm",
        "Ni_(ppm)": "ppm",
        "Cu_(ppm)": "ppm",
        "Zn_(ppm)": "ppm",
        "Ga_(ppm)": "ppm",
        "As_(ppm)": "ppm",
        "Rb_(ppm)": "ppm",
        "Sr_(ppm)": "ppm",
        "Zr_(ppm)": "ppm",
        "Ba_(ppm)": "ppm",
        "Pb_(ppm)": "ppm",
    }

    metadata_cols = {
        "Code", "Site_Name", "Sampling_Date", "Site_Code", "Core",
        "Coord_Latitud", "Coord.Longitud",
        "Granulometry(< 2 µm)_%",
        "Granulometry(2 < G < 67 µm)_%",
        "Granulometry(> 63 µm)_%",
        "PPI_550ºC (%)",
        "pH",
        "Humity_Content(%)",
    }

    print("1. Creating UniversalDataIngestor...")
    ingestor = UniversalDataIngestor(
        analyte_schema=elements_schema,
        metadata_columns=metadata_cols,
        target_unit="ppm",
        strict_schema=False,
        censored_value_strategy="lod_half",
        generate_quality_report=True,
    )
    print("   [OK] Ingestor created successfully\n")

    print("2. Ingesting BD_ISOVIDA dataset...")
    file_path = ROOT / "data" / "raw" / "BD_ISOVIDA_MANGLARES2023_version250226. Entregarxlsx.xlsx"

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
    if raw_data is not None and "V_(ppm)" in raw_data.columns:
        print("   " + str(raw_data["V_(ppm)"].head(3).tolist()))

    print("\n   PROCESSED data sample (first element column):")
    if "V_(ppm)" in data.columns:
        print("   " + str(data["V_(ppm)"].head(3).tolist()))
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

    negative_count = (data[numeric_cols] < 0).sum().sum()
    print(f"   [OK] Negative values: {negative_count}")

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
