from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"features": [1, 2, 3, 4, 5]})
    assert response.status_code == 200
    assert "prediction" in response.json()
