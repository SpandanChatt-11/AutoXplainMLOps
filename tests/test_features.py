import numpy as np
import pandas as pd
import pytest

from src.features import (
    parse_price,
    parse_km_driven,
    compute_car_age,
    parse_owner_number,
    split_brand_model,
    engineer_core_features,
    encode_categorical,
)


class TestParsePrice:
    def test_lakh(self):
        assert parse_price("8.65L") == 865000.0

    def test_crore(self):
        assert parse_price("1.2Cr") == 12_000_000.0

    def test_thousand(self):
        # the exact bug caught during the real scrape - "90K" style listings
        assert parse_price("90K") == 90000.0

    def test_missing(self):
        assert np.isnan(parse_price(np.nan))

    def test_unparseable(self):
        assert np.isnan(parse_price("not a price"))


class TestParseKmDriven:
    def test_with_commas_and_suffix(self):
        result = parse_km_driven(pd.Series(["65,000 kms", "1,10,000 kms"]))
        assert result.tolist() == [65000.0, 110000.0]


class TestComputeCarAge:
    def test_basic(self):
        result = compute_car_age(pd.Series([2020, 2015]), current_year=2026)
        assert result.tolist() == [6, 11]


class TestParseOwnerNumber:
    def test_ordinal_extraction(self):
        result = parse_owner_number(pd.Series(["1st Owner", "2nd Owner", "0th Owner"]))
        assert result.tolist() == [1.0, 2.0, 0.0]


class TestSplitBrandModel:
    def test_single_word_brand(self):
        brand, model = split_brand_model("Hyundai Creta SX")
        assert brand == "Hyundai"
        assert model == "Creta SX"

    def test_multi_word_brand(self):
        brand, model = split_brand_model("Maruti Suzuki Dzire ZXI Plus")
        assert brand == "Maruti Suzuki"
        assert model == "Dzire ZXI Plus"

    def test_single_word_title(self):
        brand, model = split_brand_model("Hyundai")
        assert brand == "Hyundai"
        assert model == ""


class TestEngineerCoreFeatures:
    def test_end_to_end_on_small_frame(self):
        df = pd.DataFrame({
            "Price": ["8.65L", "90K"],
            "KM Driven": ["45,000 kms", "1,20,000 kms"],
            "Year": [2020, 2015],
            "Owner": ["1st Owner", "2nd Owner"],
            "Title": ["Maruti Suzuki Dzire ZXI", "Hyundai i10 Magna"],
        })
        result = engineer_core_features(df, current_year=2026)
        assert result["Price_INR"].tolist() == [865000.0, 90000.0]
        assert result["KM_Driven"].tolist() == [45000.0, 120000.0]
        assert result["Car_Age"].tolist() == [6, 11]
        assert result["Owner_Number"].tolist() == [1.0, 2.0]
        assert result["Brand"].tolist() == ["Maruti Suzuki", "Hyundai"]
        assert result["Model"].tolist() == ["Dzire ZXI", "i10 Magna"]


class TestEncodeCategorical:
    def test_drop_first_baseline_is_deterministic(self):
        # Alphabetically: CNG < Diesel < Petrol -> CNG should be the dropped baseline
        df = pd.DataFrame({
            "Fuel Type": ["CNG", "Diesel", "Petrol", "Diesel"],
            "Transmission": ["Automatic", "Manual", "Manual", "Automatic"],
            "Brand": ["Honda", "Hyundai", "Honda", "Hyundai"],
            "Price_INR": [100, 200, 300, 400],
        })
        df_ml, metadata = encode_categorical(df, top_n_brands=12)

        assert "Fuel_CNG" not in df_ml.columns  # dropped baseline
        assert "Fuel_Diesel" in df_ml.columns
        assert "Fuel_Petrol" in df_ml.columns

        assert "Trans_Automatic" not in df_ml.columns  # dropped baseline
        assert "Trans_Manual" in df_ml.columns

        assert metadata["fuel_categories"] == ["CNG", "Diesel", "Petrol"]
        assert metadata["transmission_categories"] == ["Automatic", "Manual"]

    def test_metadata_records_top_brands(self):
        df = pd.DataFrame({
            "Fuel Type": ["Petrol"] * 5,
            "Transmission": ["Manual"] * 5,
            "Brand": ["Honda", "Honda", "Toyota", "Toyota", "RareBrand"],
            "Price_INR": [100, 200, 300, 400, 500],
        })
        df_ml, metadata = encode_categorical(df, top_n_brands=2)
        assert set(metadata["top_brands"]) == {"Honda", "Toyota"}
        assert "Brand_Other" in df_ml.columns  # RareBrand should be bucketed
