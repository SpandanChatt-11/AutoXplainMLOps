"""
Full data pipeline: raw scraped CSV -> cleaned -> feature-engineered -> geocoded
-> one-hot encoded ML-ready CSV.

Run directly: python -m src.build_dataset
"""

import os
import joblib

from src.data_prep import clean_pipeline
from src.features import engineer_core_features, geocode_locations, encode_categorical


def real_geocode_fn():
    """Returns a geopy-Nominatim-backed geocode function for production use.
    Kept as a separate factory (rather than imported directly at module level)
    so tests never accidentally require network access or the geopy dependency
    to import this module."""
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="autoxplain")

    def geocode(query):
        result = geolocator.geocode(query)
        return (result.latitude, result.longitude) if result else None

    return geocode


def build_dataset(
    raw_csv_path: str,
    output_csv_path: str,
    metadata_output_path: str = "models/category_metadata.pkl",
    use_real_geocoding: bool = True,
):
    df = clean_pipeline(raw_csv_path)
    df = engineer_core_features(df)

    geocode_fn = real_geocode_fn() if use_real_geocoding else None
    df = geocode_locations(df, geocode_fn=geocode_fn)

    df_ml, metadata = encode_categorical(df)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_ml.to_csv(output_csv_path, index=False)

    os.makedirs(os.path.dirname(metadata_output_path), exist_ok=True)
    joblib.dump(metadata, metadata_output_path)

    print(f"Built ML-ready dataset: {df_ml.shape} -> {output_csv_path}")
    print(f"Saved category metadata -> {metadata_output_path}")
    return df_ml, metadata


if __name__ == "__main__":
    import sys

    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/cardekho_kolkata_scraped_data.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/cardekho_kolkata_ml_ready.csv"
    build_dataset(raw_path, out_path)
