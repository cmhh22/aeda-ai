"""
Test the AEDA Ingestion Module with the BD_ISOVIDA dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from src.ingestion.universal_data_ingestor import UniversalDataIngestor

    print("\n" + "=" * 80)
    print("AEDA FRAMEWORK - INGESTION MODULE TEST")
    print("Dataset: BD_ISOVIDA Manglares")
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
        "As_(ppm)": "ppm",
        "Pb_(ppm)": "ppm",
    }

    print("1. Creating ingestor...")
    ingestor = UniversalDataIngestor(
        analyte_schema=elements_schema,
        target_unit="ppm",
        censored_value_strategy="lod_half",
        generate_quality_report=True,
    )
    print("   OK - Ingestor created\n")

    print("2. Loading data...")
    file_path = ROOT / "data" / "raw" / "BD_ISOVIDA_MANGLARES2023_version250226. Entregarxlsx.xlsx"

    try:
        result = ingestor.run(str(file_path), matrix_type_hint="sediment")
        print("   OK - Data loaded\n")
    except Exception as error:
        print(f"   ERROR: {error}")
        sys.exit(1)

    print("3. Results:")
    data = result["data"]
    metadata = result["metadata"]

    print(f"   - Samples: {data.shape[0]}")
    print(f"   - Variables: {data.shape[1]}")
    print(f"   - Matrix type: {metadata.get('matrix_type_detected', 'unknown')}")
    print(f"   - Confidence: {metadata.get('matrix_type_confidence', 0):.1%}\n")

    print("4. Censored values:")
    censored_summary = result.get("censored_summary", {})
    print(f"   - BDL handled: {censored_summary.get('total_bdl_values', 0)}")
    print(f"   - AQL handled: {censored_summary.get('total_aloq_values', 0)}")
    print(f"   - Strategy: {censored_summary.get('strategy_used', 'N/A')}\n")

    print("5. Quality Report:")
    quality_report = result.get("quality_report", "")
    if quality_report:
        report_path = ROOT / "data" / "processed" / "ingestion_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(quality_report, encoding="utf-8")
        print(f"   - Saved to: {report_path}\n")

        lines = quality_report.split("\n")[:20]
        for line in lines:
            print("     " + line)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE - SUCCESS")
    print("=" * 80 + "\n")
