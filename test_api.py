"""
Module: test_api.py
Description: Automated test suite for FastAPI /predict and /metrics endpoints,
             verifying Pydantic schema validation with valid and invalid payloads.
"""

import json
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_metrics_endpoint():
    """Test the /metrics endpoint returns expected model metrics."""
    response = client.get("/metrics")
    print(f"\n[Test 1] Testing /metrics endpoint -> Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    print("Metrics response:")
    print(json.dumps(data, indent=2))
    assert "metrics" in data
    assert "accuracy" in data["metrics"]
    assert "roc_auc" in data["metrics"]
    print("✓ /metrics endpoint test passed!")

def test_predict_valid_payload():
    """Test /predict with valid customer payload."""
    valid_payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 159.70
    }
    print(f"\n[Test 2] Testing /predict with VALID payload...")
    response = client.post("/predict", json=valid_payload)
    print(f"Status Code: {response.status_code}")
    print("Prediction Result:", response.json())
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    res = response.json()
    assert "churn_prediction" in res
    assert "churn_probability" in res
    assert res["churn_label"] in ["Yes", "No"]
    print("✓ Valid payload prediction test passed!")

def test_predict_invalid_negative_tenure():
    """Test /predict with invalid negative tenure (should fail validation)."""
    invalid_payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": -10,  # INVALID: negative tenure
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 45.0,
        "TotalCharges": 450.0
    }
    print(f"\n[Test 3] Testing /predict with INVALID payload (negative tenure: -10)...")
    response = client.post("/predict", json=invalid_payload)
    print(f"Status Code: {response.status_code} (Expected 422)")
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    error_details = response.json()
    print("Pydantic Validation Error Details:", json.dumps(error_details, indent=2))
    assert any("tenure" in str(err) for err in error_details["detail"])
    print("✓ Negative tenure rejected with 422 as expected!")

def test_predict_invalid_contract_category():
    """Test /predict with invalid categorical contract value."""
    invalid_payload = {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Lifetime Agreement",  # INVALID: not in allowed Literal values
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 99.0,
        "TotalCharges": 1188.0
    }
    print(f"\n[Test 4] Testing /predict with INVALID contract category ('Lifetime Agreement')...")
    response = client.post("/predict", json=invalid_payload)
    print(f"Status Code: {response.status_code} (Expected 422)")
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    error_details = response.json()
    print("Pydantic Validation Error Details:", json.dumps(error_details, indent=2))
    assert any("Contract" in str(err) for err in error_details["detail"])
    print("✓ Invalid contract value rejected with 422 as expected!")

def test_predict_missing_required_field():
    """Test /predict with missing required field (MonthlyCharges)."""
    invalid_payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        # MonthlyCharges is intentionally missing
        "TotalCharges": 150.0
    }
    print(f"\n[Test 5] Testing /predict with MISSING required field (MonthlyCharges)...")
    response = client.post("/predict", json=invalid_payload)
    print(f"Status Code: {response.status_code} (Expected 422)")
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"
    print("✓ Missing field rejected with 422 as expected!")

def run_all_tests():
    print("=" * 60)
    print(">>> RUNNING FASTAPI ENDPOINT VALIDATION TESTS <<<")
    print("=" * 60)
    test_metrics_endpoint()
    test_predict_valid_payload()
    test_predict_invalid_negative_tenure()
    test_predict_invalid_contract_category()
    test_predict_missing_required_field()
    print("\n" + "=" * 60)
    print(">>> ALL 5 FASTAPI API VALIDATION TESTS PASSED! <<<")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
