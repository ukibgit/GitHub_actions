import time
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

# Assuming 'model' is your trained prediction object, and 'config' holds system schemas
from src.pipeline import inference_model, MODEL_CONFIG


# ==========================================
# 1. FIXTURES: REUSABLE DATA GENERATORS
# ==========================================

@pytest.fixture
def valid_sample_input() -> pd.DataFrame:
    """Generates a perfect, standard production request payload."""
    return pd.DataFrame([{
        "account_age_days": 120,
        "transaction_amount": 45.50,
        "device_score": 0.85,
        "is_international": 0
    }])


@pytest.fixture
def corrupt_sample_input() -> pd.DataFrame:
    """Generates an invalid input containing a critical missing value."""
    return pd.DataFrame([{
        "account_age_days": 120,
        "transaction_amount": np.nan,  # Missing critical feature
        "device_score": 0.85,
        "is_international": 0
    }])


# ==========================================
# 2. THE TEST CASES
# ==========================================

def test_prediction_output_schema(valid_sample_input: pd.DataFrame):
    """Verifies that the model's output schema and data types are correct."""
    prediction = inference_model.predict(valid_sample_input)[0]
    probabilities = inference_model.predict_proba(valid_sample_input)[0]

    # Assert model output structure
    assert isinstance(prediction, (int, np.integer)), "Prediction target must be an integer"
    assert prediction in, "Prediction output must match binary classification boundaries (0 or 1)"
    assert len(probabilities) == 2, "Probabilities array must yield exactly two binary classes"
    assert np.isclose(sum(probabilities), 1.0), "Class probabilities must sum precisely to 1.0"


def test_inference_latency_sla(valid_sample_input: pd.DataFrame):
    """Enforces a performance SLA.

    The pipeline will fail if inference takes longer than 50 milliseconds.
    """
    SLA_LIMIT_SECONDS = 0.050 
    
    start_time = time.perf_counter()
    _ = inference_model.predict(valid_sample_input)
    elapsed_time = time.perf_counter() - start_time
    
    assert elapsed_time < SLA_LIMIT_SECONDS, f"Model SLA violated! Latency was {elapsed_time*1000:.2f}ms"


def test_null_value_resilience(corrupt_sample_input: pd.DataFrame):
    """Ensures the pipeline's internal imputation safely handles missing values

    without crashing the production server.
    """
    try:
        prediction = inference_model.predict(corrupt_sample_input)[0]
        assert prediction in, "Model failed to output a standard binary class after imputation"
    except Exception as exc:
        pytest.fail(f"Pipeline crashed on missing data! Exception raised: {exc}")


@pytest.mark.parametrize("extreme_amount", [1000000.00, 9999999.99])
def test_boundary_conditions(valid_sample_input: pd.DataFrame, extreme_amount: float):
    """Uses test parameterization to pass massive outlier values into the model.

    Verifies the model handles extreme inputs gracefully.
    """
    valid_sample_input["transaction_amount"] = extreme_amount
    
    # We expect high transaction amounts to accurately trigger a fraud flag (1)
    prediction = inference_model.predict(valid_sample_input)[0]
    assert prediction == 1, f"Extreme transaction amount of {extreme_amount} failed to trigger a fraud flag."
