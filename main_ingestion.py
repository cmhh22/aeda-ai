"""
AEDA Framework - Main Entry Point for Ingestion Module
Example usage for Module 1 - Universal Environmental Data Ingestion
"""

from __future__ import annotations

import sys
from pathlib import Path

# Setup path
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.ingestion.universal_data_ingestor import UniversalDataIngestor


def main_ingestion_example():
    """
    Example for the universal ingestion module.
    Processes the BD_ISOVIDA mangrove sediment dataset.
    """
    
    print("\n" + "="*80)
    print("AEDA FRAMEWORK - MODULE 1: UNIVERSAL INGESTION")
    print("="*80)
    
    # Define schema of elements from FRX analysis
    elements_schema = {
        "Na_(%)": "%",
        "Mg_(%)": "%",
        "Al_(%)": "%",
        "Si_(%)": "%",
        "P_(ppm)": "ppm",
        "S_(ppm)": "ppm",
        "Cl_(ppm)": "ppm",
        "K_(%)": "%",
        "Ca_(%)": "%",
        "Sc_(ppm)": "ppm",
        "Ti_(ppm)": "ppm",
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
        "Br_(ppm)": "ppm",
        "Rb_(ppm)": "ppm",
        "Sr_(ppm)": "ppm",
        "Y_(ppm)": "ppm",
        "Zr_(ppm)": "ppm",
        "Nb_(ppm)": "ppm",
        "Mo_(ppm)": "ppm",
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
    
    # Create ingestor
    print("\n1. Initializing universal ingestor...")
    ingestor = UniversalDataIngestor(
        analyte_schema=elements_schema,
        metadata_columns=metadata_cols,
        target_unit="ppm",
        strict_schema=False,
        censored_value_strategy="lod_half",
        generate_quality_report=True,
    )
    print("   OK - Ingestor ready\n")
    
    # Process data
    print("2. Processing BD_ISOVIDA data...")
    file_path = "data/raw/BD_ISOVIDA_MANGLARES2023_version250226. Entregarxlsx.xlsx"
    
    try:
        result = ingestor.run(
            file_path,
            matrix_type_hint="sediment",
        )
        print("   OK - Data processed\n")
    except FileNotFoundError:
        print(f"   ERROR: File not found: {file_path}")
        print("   Please verify that the file exists at: data/raw/\n")
        return None
    except Exception as e:
        print(f"   ERROR: {e}\n")
        return None
    
    # Show results
    print("3. Ingestion results:")
    data = result["data"]
    metadata = result["metadata"]
    
    print(f"   - Samples: {data.shape[0]}")
    print(f"   - Variables: {data.shape[1]}")
    print(f"   - Detected matrix: {metadata.get('matrix_type_detected', 'unknown')}")
    print(f"   - Detection confidence: {metadata.get('matrix_type_confidence', 0):.1%}\n")
    
    # Show censored values
    print("4. Censored value handling:")
    censored_summary = result.get("censored_summary", {})
    if censored_summary:
        print(f"   - BDL values processed: {censored_summary.get('total_bdl_values', 0)}")
        print(f"   - ALOQ values processed: {censored_summary.get('total_aloq_values', 0)}")
        print(f"   - Strategy: {censored_summary.get('strategy_used', 'N/A')}\n")
    
    # Save results
    print("5. Saving processed data...")
    output_path = Path("data/processed") / "BD_ISOVIDA_processed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"   - Data: {output_path}\n")
    
    # Save quality report
    quality_report = result.get("quality_report", "")
    if quality_report:
        report_path = Path("data/processed") / "ingestion_quality_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(quality_report)
        print(f"   - Quality report: {report_path}\n")
    
    print("="*80)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nData is ready for Module 2 (Exploration)")
    print("See docs/INGESTION_MODULE_GUIDE.md for more details\n")
    
    return result


if __name__ == "__main__":
    result = main_ingestion_example()
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)
