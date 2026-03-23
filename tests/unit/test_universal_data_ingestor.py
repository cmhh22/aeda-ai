from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingestion.raw_data_ingestor import RawDataIngestor
from src.ingestion.universal_data_ingestor import UniversalDataIngestor


def test_raw_data_ingestor_parses_frx_notation(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "As": ["< 17", "25"],
            "Pb": ["123 ± 5", "> 12000 (11873)"],
        }
    )
    file_path = tmp_path / "frx_raw.csv"
    data.to_csv(file_path, index=False)

    ingestor = RawDataIngestor(chemical_schema={"As": "ppm", "Pb": "ppm"}, target_unit="ppm")
    result = ingestor.run(file_path)

    assert "data" in result
    parsed = result["data"]

    assert float(parsed.loc[0, "As"]) == 17.0
    assert float(parsed.loc[0, "Pb"]) == 123.0
    assert float(parsed.loc[1, "Pb"]) == 11873.0
    assert "U_Pb" in parsed.columns
    assert float(parsed.loc[0, "U_Pb"]) == 5.0

    flags = result["metadata"]["ingestion"]["quality_flags"]
    assert flags["As"]["lt_lod"] == 1
    assert flags["Pb"]["gt_limit"] == 1
    assert flags["Pb"]["has_uncertainty"] == 1


def test_universal_ingestor_detects_sediment_matrix(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "As": [12, 14],
            "Pb": [33, 35],
            "Depth": [5, 10],
            "PPI": [8.1, 7.9],
            "Silt": [22.0, 19.5],
        }
    )
    file_path = tmp_path / "sediment.csv"
    data.to_csv(file_path, index=False)

    ingestor = UniversalDataIngestor(analyte_schema={"As": "ppm", "Pb": "ppm"})
    result = ingestor.run(file_path)

    assert result["matrix_type_detected"] == "sediment"
    assert "data" in result
    assert not result["data"].empty
