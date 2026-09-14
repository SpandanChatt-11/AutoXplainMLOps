import pytest

from app.app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


class TestRoutes:
    def test_get_index_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"AutoXplain" in response.data

    def test_get_index_has_all_dropdowns(self, client):
        response = client.get("/")
        html = response.data.decode()
        assert '<select name="brand">' in html
        assert '<select name="fuel">' in html
        assert '<select name="transmission">' in html

    def test_post_valid_form_returns_price_and_range(self, client):
        response = client.post("/", data={
            "brand": "Hyundai", "fuel": "Petrol", "transmission": "Manual",
            "car_age": "5", "km_driven": "45000", "owner_number": "1",
            "latitude": "22.5726", "longitude": "88.3639",
        })
        assert response.status_code == 200
        html = response.data.decode()
        assert "Predicted Price" in html
        assert "base64," in html  # the embedded LIME image

    def test_post_baseline_categories_does_not_error(self, client):
        # CNG / Automatic / a brand outside the explicit dummy list - all fall
        # through as the correct all-zero baseline encoding, should not error
        response = client.post("/", data={
            "brand": "Other", "fuel": "CNG", "transmission": "Automatic",
            "car_age": "2", "km_driven": "10000", "owner_number": "1",
            "latitude": "22.57", "longitude": "88.36",
        })
        assert response.status_code == 200
        assert b"Predicted Price" in response.data
