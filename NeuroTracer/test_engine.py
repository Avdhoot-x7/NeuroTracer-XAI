import pytest
import sys
import os
from fastapi.testclient import TestClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NeuroTracer.main import app

client = TestClient(app)

# Test 1: Verify the Engine Logic directly
def test_hallucination_logic():
    from NeuroTracer.engine import NeuroTracer
    tracer = NeuroTracer()
    
    fact = tracer.analyze("The capital of", "The capital of France is")
    hallucination = tracer.analyze("The capital of", "The capital of Mars is")
    
    # Assertions: This is the "Proof"
    assert hallucination["risk_score"] > fact["risk_score"]
    assert fact["is_hallucination"] is False
    assert hallucination["is_hallucination"] is True

# Test 2: Verify the API Endpoint
def test_api_endpoint():
    payload = {
        "base_prompt": "The capital of",
        "claim_prompt": "The capital of France is",
        "top_k": 50
    }
    response = client.post("/trace", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "expert_location" in data