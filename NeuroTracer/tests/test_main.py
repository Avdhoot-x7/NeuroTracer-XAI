from fastapi.testclient import TestClient
from main import app # This imports your FastAPI app

client = TestClient(app)

def test_read_main():
    # Simple test to check if the API is alive
    response = client.get("/docs")
    assert response.status_code == 200

def test_analyze_endpoint():
    # Tests your specific logic
    payload = {
        "base": "The capital of",
        "claim": "The capital of France is Paris"
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200