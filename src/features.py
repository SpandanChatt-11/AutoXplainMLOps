"""
Feature engineering: cleaned dataframe -> ML-ready dataframe.

Mirrors the logic validated in the original preprocessing notebook, extracted into
testable functions.
"""

import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

MULTI_WORD_BRANDS = ["Maruti Suzuki", "Mercedes-Benz", "Land Rover", "Mahindra & Mahindra"]

KOLKATA_CENTER = (22.5726, 88.3639)


def parse_price(val) -> float:
    """'8.65L' -> 865000.0, '1.2Cr' -> 12000000.0, '90K' -> 90000.0"""
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    match = re.match(r"([\d.]+)\s*(L|Cr|K)", val, re.IGNORECASE)
    if not match:
        return np.nan
    number, unit = match.groups()
    number = float(number)
    multiplier = {"K": 1_000, "L": 100_000, "CR": 10_000_000}[unit.upper()]
    return number * multiplier


def parse_km_driven(series: pd.Series) -> pd.Series:
    """'65,000 kms' -> 65000.0"""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )


def compute_car_age(year_series: pd.Series, current_year: int = None) -> pd.Series:
    if current_year is None:
        current_year = datetime.now().year
    return current_year - year_series


def parse_owner_number(owner_series: pd.Series) -> pd.Series:
    """'1st Owner' -> 1.0, '2nd Owner' -> 2.0"""
    return owner_series.astype(str).str.extract(r"(\d+)")[0].astype(float)


def split_brand_model(title: str):
    """'Maruti Suzuki Dzire ZXI' -> ('Maruti Suzuki', 'Dzire ZXI')"""
    for brand in MULTI_WORD_BRANDS:
        if title.startswith(brand):
            return brand, title[len(brand):].strip()
    parts = title.split(maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], ""


def engineer_core_features(df: pd.DataFrame, current_year: int = None) -> pd.DataFrame:
    """Adds Price_INR, KM_Driven, Car_Age, Owner_Number, Brand, Model to a cleaned dataframe."""
    df = df.copy()
    df["Price_INR"] = df["Price"].apply(parse_price)
    df["KM_Driven"] = parse_km_driven(df["KM Driven"])
    df["Car_Age"] = compute_car_age(df["Year"], current_year=current_year)
    df["Owner_Number"] = parse_owner_number(df["Owner"])

    brand_model = df["Title"].apply(split_brand_model)
    df["Brand"] = brand_model.apply(lambda x: x[0])
    df["Model"] = brand_model.apply(lambda x: x[1])
    return df


def geocode_locations(df: pd.DataFrame, geocode_fn=None, sleep_seconds: float = 1.0) -> pd.DataFrame:
    """Adds Latitude/Longitude by geocoding unique Location values.

    `geocode_fn` is injectable so tests (and CI, which has no network access to
    Nominatim) can pass a fake/cached lookup instead of hitting a live geocoding
    service. In production, pass a real geopy-backed function, e.g.:

        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="autoxplain")
        def real_geocode(query):
            result = geolocator.geocode(query)
            return (result.latitude, result.longitude) if result else None
    """
    df = df.copy()
    unique_locations = df["Location"].unique()
    location_coords = {}

    for loc in unique_locations:
        if geocode_fn is None:
            location_coords[loc] = None
        else:
            try:
                location_coords[loc] = geocode_fn(f"{loc}, India")
            except Exception:
                location_coords[loc] = None
            time.sleep(sleep_seconds)

    df["Latitude"] = df["Location"].map(lambda x: (location_coords[x] or KOLKATA_CENTER)[0])
    df["Longitude"] = df["Location"].map(lambda x: (location_coords[x] or KOLKATA_CENTER)[1])
    return df


def encode_categorical(df: pd.DataFrame, top_n_brands: int = 12, top_brands=None):
    """One-hot encodes Fuel Type, Transmission, and a top-N-bucketed Brand.

    Returns (encoded_df, metadata) where metadata records exactly which categories
    were used, so inference-time code can reconstruct the same encoding scheme
    without guessing at dropped baseline categories.
    """
    df = df.copy()

    if top_brands is None:
        top_brands = df["Brand"].value_counts().head(top_n_brands).index.tolist()

    df["Brand_grouped"] = df["Brand"].where(df["Brand"].isin(top_brands), "Other")

    fuel_categories = sorted(df["Fuel Type"].dropna().unique().tolist())
    transmission_categories = sorted(df["Transmission"].dropna().unique().tolist())
    brand_categories = sorted(df["Brand_grouped"].dropna().unique().tolist())

    df_ml = pd.get_dummies(
        df,
        columns=["Fuel Type", "Transmission", "Brand_grouped"],
        prefix=["Fuel", "Trans", "Brand"],
        drop_first=True,
    )

    drop_cols = [
        c
        for c in ["Title", "Link", "Price", "KM Driven", "Year", "Owner", "Location",
                   "EMI", "EMI_INR", "Brand", "Model"]
        if c in df_ml.columns
    ]
    df_ml = df_ml.drop(columns=drop_cols)

    metadata = {
        "fuel_categories": fuel_categories,
        "transmission_categories": transmission_categories,
        "brand_categories": brand_categories,
        "top_brands": top_brands,
        "feature_names": [c for c in df_ml.columns if c != "Price_INR"],
    }
    return df_ml, metadata
