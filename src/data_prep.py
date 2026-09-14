"""
Data cleaning: raw scraped CSV -> cleaned dataframe (before feature engineering).

Mirrors the logic validated in the original preprocessing notebook, extracted into
testable functions.
"""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe on Link - the one column guaranteed to uniquely identify a listing,
    even if two different ads happen to share identical text elsewhere."""
    return df.drop_duplicates(subset="Link").reset_index(drop=True)


def drop_unneeded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that add no modeling value: marketing text, and the
    Price/KM Driven detail-page fallback columns (redundant once the main
    listing-card columns are complete)."""
    cols_to_drop = [
        c
        for c in ["AI Highlights", "Price (detail page)", "KM Driven (detail page)"]
        if c in df.columns
    ]
    return df.drop(columns=cols_to_drop)


def fill_fuel_and_transmission(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce Fuel Type / Transmission with their detail-page counterparts
    (which are near-complete), then drop the now-redundant fallback columns."""
    df = df.copy()
    if "Fuel Type (detail page)" in df.columns:
        df["Fuel Type"] = df["Fuel Type"].fillna(df["Fuel Type (detail page)"])
        df = df.drop(columns=["Fuel Type (detail page)"])
    if "Transmission (detail page)" in df.columns:
        df["Transmission"] = df["Transmission"].fillna(df["Transmission (detail page)"])
        df = df.drop(columns=["Transmission (detail page)"])
    return df


def fill_location(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Location with the mode - reasonable at a low missingness ratio."""
    df = df.copy()
    if df["Location"].isna().any():
        mode_value = df["Location"].mode()[0]
        df["Location"] = df["Location"].fillna(mode_value)
    return df


def drop_negligible_missing_rows(df: pd.DataFrame, columns=("Owner",)) -> pd.DataFrame:
    """Drop rows missing a value in low-missingness categorical columns where no
    sensible default exists (e.g. Owner) - safer than inventing a category."""
    cols = [c for c in columns if c in df.columns]
    return df.dropna(subset=cols).reset_index(drop=True)


def clean_pipeline(raw_csv_path: str) -> pd.DataFrame:
    """Full cleaning pipeline: raw scraped CSV -> cleaned dataframe, ready for
    feature engineering (features.py)."""
    df = load_raw_data(raw_csv_path)
    df = deduplicate(df)
    df = drop_unneeded_columns(df)
    df = fill_fuel_and_transmission(df)
    df = fill_location(df)
    df = drop_negligible_missing_rows(df, columns=["Owner"])
    return df
