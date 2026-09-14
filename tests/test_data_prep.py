import numpy as np
import pandas as pd

from src.data_prep import (
    deduplicate,
    drop_unneeded_columns,
    fill_fuel_and_transmission,
    fill_location,
    drop_negligible_missing_rows,
)


class TestDeduplicate:
    def test_dedupes_on_link_not_full_row(self):
        df = pd.DataFrame({
            "Link": ["a.com/1", "a.com/1", "a.com/2"],
            "Title": ["Car A", "Car A (slightly different scrape)", "Car B"],
        })
        result = deduplicate(df)
        assert len(result) == 2
        assert result["Link"].tolist() == ["a.com/1", "a.com/2"]


class TestDropUnneededColumns:
    def test_drops_expected_columns_when_present(self):
        df = pd.DataFrame({
            "AI Highlights": ["x"], "Price (detail page)": ["y"],
            "KM Driven (detail page)": ["z"], "Price": ["8L"],
        })
        result = drop_unneeded_columns(df)
        assert "AI Highlights" not in result.columns
        assert "Price (detail page)" not in result.columns
        assert "Price" in result.columns

    def test_no_error_when_columns_absent(self):
        df = pd.DataFrame({"Price": ["8L"]})
        result = drop_unneeded_columns(df)
        assert list(result.columns) == ["Price"]


class TestFillFuelAndTransmission:
    def test_coalesces_from_detail_page(self):
        df = pd.DataFrame({
            "Fuel Type": ["Petrol", np.nan],
            "Fuel Type (detail page)": ["Petrol", "Diesel"],
            "Transmission": [np.nan, "Manual"],
            "Transmission (detail page)": ["Automatic", "Manual"],
        })
        result = fill_fuel_and_transmission(df)
        assert result["Fuel Type"].tolist() == ["Petrol", "Diesel"]
        assert result["Transmission"].tolist() == ["Automatic", "Manual"]
        assert "Fuel Type (detail page)" not in result.columns
        assert "Transmission (detail page)" not in result.columns


class TestFillLocation:
    def test_fills_with_mode(self):
        df = pd.DataFrame({"Location": ["Salt Lake", "Salt Lake", np.nan]})
        result = fill_location(df)
        assert result["Location"].tolist() == ["Salt Lake", "Salt Lake", "Salt Lake"]


class TestDropNegligibleMissingRows:
    def test_drops_rows_missing_owner(self):
        df = pd.DataFrame({"Owner": ["1st Owner", np.nan, "2nd Owner"]})
        result = drop_negligible_missing_rows(df, columns=["Owner"])
        assert len(result) == 2
        assert result["Owner"].tolist() == ["1st Owner", "2nd Owner"]
