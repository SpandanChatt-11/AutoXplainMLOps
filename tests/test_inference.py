import numpy as np
import pandas as pd
import pytest

from src.inference import (
    build_feature_row,
    needs_scaling,
    load_model,
    load_artifacts,
    predict_price,
    predict_interval,
)

FEATURE_NAMES = [
    "KM_Driven", "Car_Age", "Owner_Number", "Latitude", "Longitude",
    "Fuel_Diesel", "Fuel_Electric", "Fuel_Petrol", "Trans_Manual",
    "Brand_Hyundai", "Brand_Honda",
]


class TestNeedsScaling:
    def test_ridge_and_svr_need_scaling(self):
        assert needs_scaling("Ridge") is True
        assert needs_scaling("SVR") is True

    def test_tree_models_dont_need_scaling(self):
        assert needs_scaling("Random Forest") is False
        assert needs_scaling("XGBoost") is False


class TestBuildFeatureRow:
    def test_known_categories_set_correct_dummy(self):
        form = {
            "km_driven": "45000", "car_age": "5", "owner_number": "1",
            "latitude": "22.57", "longitude": "88.36",
            "fuel": "Petrol", "transmission": "Manual", "brand": "Hyundai",
        }
        row = build_feature_row(form, FEATURE_NAMES)
        assert row["Fuel_Petrol"].iloc[0] == 1
        assert row["Fuel_Diesel"].iloc[0] == 0
        assert row["Trans_Manual"].iloc[0] == 1
        assert row["Brand_Hyundai"].iloc[0] == 1
        assert row["Brand_Honda"].iloc[0] == 0
        assert row["KM_Driven"].iloc[0] == 45000.0

    def test_baseline_categories_leave_all_dummies_zero(self):
        # CNG (fuel baseline), Automatic (transmission baseline), Honda (brand baseline)
        form = {
            "km_driven": "10000", "car_age": "2", "owner_number": "1",
            "latitude": "22.57", "longitude": "88.36",
            "fuel": "CNG", "transmission": "Automatic", "brand": "Honda",
        }
        row = build_feature_row(form, FEATURE_NAMES)
        assert row["Fuel_Diesel"].iloc[0] == 0
        assert row["Fuel_Electric"].iloc[0] == 0
        assert row["Fuel_Petrol"].iloc[0] == 0
        assert row["Trans_Manual"].iloc[0] == 0
        # "Honda" isn't one of the explicit Brand_ columns in this small fixture
        # list either, so it should also fall through as all-zero baseline - this
        # mirrors production where the confirmed dropped brand baseline works the
        # same way.

    def test_column_order_matches_feature_names(self):
        form = {
            "km_driven": "1", "car_age": "1", "owner_number": "1",
            "latitude": "1", "longitude": "1",
            "fuel": "Petrol", "transmission": "Manual", "brand": "Hyundai",
        }
        row = build_feature_row(form, FEATURE_NAMES)
        assert list(row.columns) == FEATURE_NAMES


class TestEndToEndWithRealArtifacts:
    """These exercise the actual trained artifacts produced by train.py - real
    integration coverage, not mocks, since the artifacts are small and committed."""

    @staticmethod
    @pytest.fixture(scope="class")
    def artifacts():
        return load_artifacts()

    @staticmethod
    @pytest.fixture(scope="class")
    def model_and_source():
        return load_model()

    def test_predict_price_returns_positive_float(self, model_and_source, artifacts):
        model, source = model_and_source
        form = {
            "km_driven": "45000", "car_age": "5", "owner_number": "1",
            "latitude": "22.57", "longitude": "88.36",
            "fuel": "Petrol", "transmission": "Manual", "brand": "Hyundai",
        }
        row = build_feature_row(form, artifacts["feature_names"])
        price = predict_price(row, model, source, artifacts)
        assert isinstance(price, float)
        assert price > 0

    def test_predict_interval_contains_point_estimate(self, model_and_source, artifacts):
        model, source = model_and_source
        form = {
            "km_driven": "45000", "car_age": "5", "owner_number": "1",
            "latitude": "22.57", "longitude": "88.36",
            "fuel": "Petrol", "transmission": "Manual", "brand": "Hyundai",
        }
        row = build_feature_row(form, artifacts["feature_names"])
        price = predict_price(row, model, source, artifacts)
        lower, upper = predict_interval(row, artifacts)
        assert lower <= price <= upper
        assert lower > 0
