from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import DEFAULT_NAAQS_LIMITS, DataReconstructor, DataStandardizer, OutlierDetector


def _read_input(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)

    raise ValueError("Unsupported input format. Use .csv, .xlsx, or .xls")


def _build_physical_limits(data: pd.DataFrame) -> dict[str, tuple[float | None, float | None]]:
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        raise ValueError("Input dataset has no numeric columns for preprocessing")

    limits: dict[str, tuple[float | None, float | None]] = {}
    for column in numeric_columns:
        if column in DEFAULT_NAAQS_LIMITS:
            limits[column] = DEFAULT_NAAQS_LIMITS[column]
        else:
            limits[column] = (None, None)

    return limits


def _default_output_path(input_path: Path, output_path: str | None) -> Path:
    if output_path:
        return Path(output_path)
    return ROOT / "data" / "processed" / f"{input_path.stem}_processed.csv"


def run_pipeline_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = _default_output_path(input_path=input_path, output_path=args.output)

    stages: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = []

    data = _read_input(input_path)

    if args.outliers:
        limits = _build_physical_limits(data)
        outlier_detector = OutlierDetector(
            physical_limits=limits,
            contamination=args.contamination,
            random_state=args.random_state,
        )

        def _outlier_stage(df: pd.DataFrame) -> pd.DataFrame:
            result = outlier_detector.run(df)
            return result["cleaned_data"]

        stages.append(("Outlier detection", _outlier_stage))

    reconstructor = DataReconstructor(random_state=args.random_state)

    def _imputation_stage(df: pd.DataFrame) -> pd.DataFrame:
        result = reconstructor.run(
            data=df,
            strategy=args.impute,
            columns=None,
            estimate_mse=False,
        )
        return result["reconstructed_data"]

    stages.append((f"Imputation ({args.impute})", _imputation_stage))

    standardizer = DataStandardizer()

    def _standardization_stage(df: pd.DataFrame) -> pd.DataFrame:
        result = standardizer.run(
            data=df,
            columns=None,
            processed_by="AEDA CLI",
            filters_applied=[f"outliers={args.outliers}", f"impute={args.impute}"],
            metadata_path=ROOT / "data" / "processed" / "metadata.json",
            dataset_name=input_path.stem,
        )
        return result["standardized_data"]

    stages.append(("Standardization", _standardization_stage))

    transformed = data.copy()
    with tqdm(total=len(stages), desc="Pipeline progress", unit="stage") as progress:
        for stage_name, stage_fn in stages:
            progress.set_postfix_str(stage_name)
            transformed = stage_fn(transformed)
            progress.update(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(output_path, index=False)

    print(f"Pipeline finished successfully. Output: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="AEDA Framework CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run-pipeline",
        help="Run preprocessing pipeline over an input dataset",
    )
    run_parser.add_argument(
        "--input",
        required=True,
        help="Input dataset path (.csv, .xlsx, .xls)",
    )
    run_parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/processed/<input>_processed.csv)",
    )
    run_parser.add_argument(
        "--impute",
        default="missforest",
        choices=["missforest", "pchip"],
        help="Imputation method",
    )
    outlier_group = run_parser.add_mutually_exclusive_group()
    outlier_group.add_argument(
        "--outliers",
        dest="outliers",
        action="store_true",
        help="Enable outlier detection stage",
    )
    outlier_group.add_argument(
        "--no-outliers",
        dest="outliers",
        action="store_false",
        help="Disable outlier detection stage",
    )
    run_parser.set_defaults(outliers=True)

    run_parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Outlier contamination ratio for Isolation Forest",
    )
    run_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    run_parser.set_defaults(handler=run_pipeline_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        handler = getattr(args, "handler")
        return handler(args)
    except Exception as error:
        print(f"Pipeline execution failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
