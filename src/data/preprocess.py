from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "books_raw_data.csv"
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "books_processed_data.csv"

EXPECTED_COLUMNS = [
    "title",
    "price",
    "availability",
    "rating",
    "category",
    "product_description",
    "upc",
]

OUTPUT_COLUMNS = [
    "title",
    "price",
    "availability",
    "rating",
    "category",
    "upc",
]


def load_raw_data(path: Path = RAW_FILE) -> pd.DataFrame:
    """Load the Phase 1 raw dataset."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the final Phase 2 cleaning decisions.

    Product descriptions are deliberately removed because they contain
    long free-text strings, encoding/mojibake problems, repeated/truncated
    prefix artefacts, and a small number of missing values. The raw field
    remains untouched in the raw dataset for traceability.
    """
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    cleaned = df.copy()

    # Remove the problematic free-text field from the modelling dataset.
    cleaned = cleaned.drop(columns=["product_description"])

    # Normalize whitespace in retained text/reference fields.
    for column in ["title", "category", "upc"]:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    # Preserve numeric types explicitly.
    cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce")
    cleaned["availability"] = pd.to_numeric(
        cleaned["availability"], errors="coerce"
    )
    cleaned["rating"] = pd.to_numeric(cleaned["rating"], errors="coerce")

    # The two missing descriptions do not cause row deletion because the
    # entire description field is intentionally excluded.
    return cleaned[OUTPUT_COLUMNS]


def validate_processed_data(df: pd.DataFrame) -> None:
    """Run Phase 2 validation checks."""
    if df.empty:
        raise ValueError("Processed dataset is empty.")

    if df.isna().any().any():
        missing = df.isna().sum()
        raise ValueError(
            f"Processed dataset contains missing values:\n{missing[missing > 0]}"
        )

    if (df["price"] <= 0).any():
        raise ValueError("Price contains non-positive values.")

    if (df["availability"] < 0).any():
        raise ValueError("Availability contains negative values.")

    if not df["rating"].between(1, 5).all():
        raise ValueError("Rating contains values outside 1-5.")

    if df["upc"].duplicated().any():
        raise ValueError("Duplicate UPC values found.")

    if df.duplicated().any():
        raise ValueError("Duplicate rows found.")


def save_processed_data(
    df: pd.DataFrame,
    path: Path = PROCESSED_FILE,
) -> None:
    """Save the cleaned Phase 2 dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("E-Commerce Pricing Analytics - Phase 2 Preprocessing")
    print("=" * 60)

    raw_df = load_raw_data()
    print(f"Raw shape: {raw_df.shape}")

    processed_df = clean_data(raw_df)
    validate_processed_data(processed_df)
    save_processed_data(processed_df)

    print(f"Processed shape: {processed_df.shape}")
    print(f"Columns: {processed_df.columns.tolist()}")
    print(f"Output saved to: {PROCESSED_FILE}")
    print("=" * 60)
    print("Phase 2 preprocessing completed successfully.")


if __name__ == "__main__":
    main()
