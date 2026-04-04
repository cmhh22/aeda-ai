"""
AEDA Framework - Ingestion Module
Universal Environmental Data Ingestion Interface

Example usage for importing heterogeneous environmental matrices (soil/sediment/water/air)
with complex analytical data (FRX, ICP-MS, etc.)
"""

from pathlib import Path
import pandas as pd
from src.ingestion.universal_data_ingestor import UniversalDataIngestor


def simple_ingestion_example():
    """
    Simplest possible usage - let the framework auto-detect everything.
    """
    # Define what elements you measured (simplified for FRX)
    elements = {
        "V": "ppm",
        "Cr": "ppm",
        "Mn": "ppm",
        "Fe": "%",
        "Co": "ppm",
        "Ni": "ppm",
        "Cu": "ppm",
        "Zn": "ppm",
        "As": "ppm",
        "Pb": "ppm",
    }
    
    # Create ingestor with defaults
    ingestor = UniversalDataIngestor(
        analyte_schema=elements,
        apply_censored_handling=False,
        generate_quality_report=True,
    )
    
    # Ingest data
    result = ingestor.run("data/raw/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx")
    
    # Access results
    print("Matrix type detected:", result["metadata"]["matrix_type_detected"])
    print("Processed data shape:", result["data"].shape)
    print("\nQuality Report:")
    print(result["extra"]["quality_report"])
    
    return result


def advanced_ingestion_with_hints():
    """
    More control - provide hints to guide matrix detection and specify imputation strategy.
    """
    elements = {
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
    
    # Create ingestor with advanced options
    ingestor = UniversalDataIngestor(
        analyte_schema=elements,
        metadata_columns=metadata_cols,
        target_unit="ppm",
        strict_schema=False,
        censored_value_strategy="lod_half",  # LOD/2 is standard in environmental science
        apply_censored_handling=False,
        generate_quality_report=True,
    )
    
    # Ingest with matrix type hint
    result = ingestor.run(
        "data/raw/BD_ISOVIDA_MANGLARES2023_rectificadaYBA_230326.xlsx",
        matrix_type_hint="sediment",
    )
    
    # Access detailed results
    data = result["data"]
    metadata = result["metadata"]
    extra = result["extra"]
    
    print(f"✓ Ingested {data.shape[0]} samples × {data.shape[1]} variables")
    print(f"✓ Matrix detected: {metadata['matrix_type_detected']}")
    print(f"  Confidence: {metadata['matrix_type_confidence']:.2%}")
    print(f"  Detection indicators: {metadata['matrix_detection_indicators']}")
    
    print(f"\n✓ Censored values handled:")
    print(f"  Strategy: {metadata['censored_value_strategy']}")
    
    censored_summary = extra.get("censored_summary", {})
    if censored_summary:
        print(f"  Total BDL values: {censored_summary.get('total_bdl_values', 0)}")
        print(f"  Total AQL values: {censored_summary.get('total_aloq_values', 0)}")
        print(f"  Columns processed: {censored_summary.get('total_columns_processed', 0)}")
    
    # Save quality report to file
    quality_report = extra.get("quality_report", "")
    if quality_report:
        report_path = "data/processed/data_ingestion_report.txt"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(quality_report)
        print(f"\n✓ Quality report saved to {report_path}")
    
    # Return processed data for further analysis
    return data, metadata, extra


def inspect_raw_vs_processed(result):
    """
    Helper function to compare raw vs processed data.
    Shows exactly what the framework did.
    """
    raw_data = result["extra"]["raw_data"]
    processed_data = result["data"]
    
    print("\n" + "="*80)
    print("RAW vs PROCESSED COMPARISON")
    print("="*80)
    
    print(f"\nRAW data sample (first 3 rows, first 5 columns):")
    print(raw_data.iloc[:3, :5])
    
    print(f"\nPROCESSED data sample (first 3 rows, first 5 columns):")
    print(processed_data.iloc[:3, :5])
    
    # Show censored value transformations
    censored_metadata = result["extra"].get("censored_metadata", [])
    if censored_metadata:
        print(f"\n✓ Censored values transformed:")
        for meta in censored_metadata[:5]:
            print(f"  {meta.column}: {meta.n_bdl} BDL + {meta.n_aloq} AQL values")


if __name__ == "__main__":
    # Run example
    print("AEDA Framework - Universal Data Ingestion")
    print("=" * 80)
    
    # Try advanced example first
    print("\n1. Advanced Ingestion with BD_ISOVIDA Dataset")
    print("-" * 80)
    
    try:
        data, metadata, extra = advanced_ingestion_with_hints()
        inspect_raw_vs_processed({"data": data, "metadata": metadata, "extra": extra})
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Using simple example instead...")
        result = simple_ingestion_example()
