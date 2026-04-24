"""
aeda.io.parsers
Environmental data ingestion module.
Supports CSV, Excel (multi-sheet), and JSON.
Automatically detects data dictionaries and metadata.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field


@dataclass
class DatasetInfo:
    """Metadata extracted automatically from the dataset."""
    n_rows: int = 0
    n_cols: int = 0
    numeric_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)
    coordinate_cols: list = field(default_factory=list)
    uncertainty_cols: list = field(default_factory=list)
    measurement_cols: list = field(default_factory=list)
    depth_col: Optional[str] = None
    site_col: Optional[str] = None
    has_dictionary: bool = False
    dictionary: Optional[pd.DataFrame] = None
    units: dict = field(default_factory=dict)
    file_format: str = ""


# Common patterns in LEA environmental datasets
KNOWN_COORDINATE_PATTERNS = ["latitud", "longitud", "lat", "lon", "x", "y"]
KNOWN_DEPTH_PATTERNS = ["profundidad", "depth", "prof"]
KNOWN_SITE_PATTERNS = ["sitio", "site", "estacion", "station", "site_name"]
KNOWN_UNCERTAINTY_PREFIX = ["u_", "inc_", "err_", "uncertainty_"]
KNOWN_DICT_SHEET_PATTERNS = ["diccionario", "dictionary", "metadata", "variables"]


def _detect_special_columns(df: pd.DataFrame) -> dict:
    """Detect special columns by name: coordinates, depth, site, uncertainty."""
    cols_lower = {c: c.lower().strip() for c in df.columns}
    result = {
        "coordinate_cols": [],
        "depth_col": None,
        "site_col": None,
        "uncertainty_cols": [],
        "measurement_cols": [],
    }

    for orig, low in cols_lower.items():
        if any(p in low for p in KNOWN_COORDINATE_PATTERNS):
            result["coordinate_cols"].append(orig)
        elif any(p in low for p in KNOWN_DEPTH_PATTERNS):
            result["depth_col"] = orig
        elif any(p in low for p in KNOWN_SITE_PATTERNS):
            result["site_col"] = orig
        elif any(low.startswith(p) for p in KNOWN_UNCERTAINTY_PREFIX):
            result["uncertainty_cols"].append(orig)

    non_special = set(result["coordinate_cols"] + result["uncertainty_cols"])
    if result["depth_col"]:
        non_special.add(result["depth_col"])
    if result["site_col"]:
        non_special.add(result["site_col"])

    for col in df.select_dtypes(include="number").columns:
        if col not in non_special:
            result["measurement_cols"].append(col)

    return result


def _detect_dictionary_sheet(sheet_names: list[str]) -> Optional[str]:
    """Find a worksheet that looks like a data dictionary."""
    for name in sheet_names:
        if any(p in name.lower() for p in KNOWN_DICT_SHEET_PATTERNS):
            return name
    return None


def _extract_units_from_dict(dict_df: pd.DataFrame) -> dict:
    """Extract a column-to-unit mapping from a data dictionary."""
    units = {}
    col_col = None
    unit_col = None

    for c in dict_df.columns:
        cl = c.lower()
        if "codigo" in cl or "columna" in cl or "variable" in cl or "column" in cl:
            col_col = c
        if "unidad" in cl or "unit" in cl:
            unit_col = c

    if col_col and unit_col:
        for _, row in dict_df.iterrows():
            if pd.notna(row[col_col]) and pd.notna(row[unit_col]):
                units[str(row[col_col]).strip()] = str(row[unit_col]).strip()

    return units


def load(
    filepath: Union[str, Path],
    sheet_name: Optional[str] = None,
) -> tuple[pd.DataFrame, DatasetInfo]:
    """
    Load an environmental data file and extract metadata automatically.

    Parameters
    ----------
    filepath : str or Path
        Path to the input file (.csv, .xlsx, .xls, .json).
    sheet_name : str, optional
        Worksheet name for Excel files. If None, the first data worksheet is used.

    Returns
    -------
    tuple[pd.DataFrame, DatasetInfo]
        DataFrame with loaded data and a DatasetInfo object with detected metadata.
    """
    filepath = Path(filepath)
    info = DatasetInfo(file_format=filepath.suffix.lower())

    if filepath.suffix.lower() in (".xlsx", ".xls"):
        xls = pd.ExcelFile(filepath)

        # Try to locate a data dictionary worksheet
        dict_sheet = _detect_dictionary_sheet(xls.sheet_names)
        if dict_sheet:
            info.has_dictionary = True
            info.dictionary = pd.read_excel(xls, sheet_name=dict_sheet)
            info.units = _extract_units_from_dict(info.dictionary)

        # Load data worksheet
        if sheet_name:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        else:
            data_sheets = [s for s in xls.sheet_names if s != dict_sheet]
            df = pd.read_excel(xls, sheet_name=data_sheets[0] if data_sheets else 0)

    elif filepath.suffix.lower() == ".csv":
        df = pd.read_csv(filepath)

    elif filepath.suffix.lower() == ".json":
        df = pd.read_json(filepath)

    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    # Detect special-purpose columns
    special = _detect_special_columns(df)

    info.n_rows, info.n_cols = df.shape
    info.numeric_cols = df.select_dtypes(include="number").columns.tolist()
    info.categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    info.coordinate_cols = special["coordinate_cols"]
    info.depth_col = special["depth_col"]
    info.site_col = special["site_col"]
    info.uncertainty_cols = special["uncertainty_cols"]
    info.measurement_cols = special["measurement_cols"]

    return df, info
